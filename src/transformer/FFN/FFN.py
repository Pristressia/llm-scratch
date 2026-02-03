import numpy as np
import numpy.typing as npt
from typing import Callable, Any
from dataclasses import dataclass
from llm_operator import layer_norm, layer_norm_backward, Layer_norm_cache, Relu, Relu_backward
from transformer import TransformerCore

ActForward = Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], Any]]
ActBackward = Callable[[npt.NDArray[np.float64], Any], npt.NDArray[np.float64]]
@dataclass
class FFN_init:

    gamma: npt.NDArray[np.float64]    # (1, 1, d_model) or (d_model, )
    beta: npt.NDArray[np.float64]     # (1, 1, d_model) or (d_model, )

    W_expand: npt.NDArray[np.float64] # (1, d_model, d_model * expand_scale) or (d_model, d_model * expand_scale)
    B_expand: npt.NDArray[np.float64] # (1, 1, d_model * expand_scale) or (d_model * expand_scale, )

    W_shrink: npt.NDArray[np.float64] # (1, d_model * expand_scale, d_model) or (d_model * expand_scale, d_model)
    B_shrink: npt.NDArray[np.float64] # (1, 1 * d_model) or (d_model, )

    activate_function: ActForward
    activate_function_backward: ActBackward
    
class FFN(TransformerCore):

    xnorm : npt.NDArray[np.float64]

    def __init__(self, init: FFN_init):
        self.gamma = init.gamma.reshape(1, 1, -1)
        self.beta = init.beta.reshape(1, 1, -1)

        self.W_expand = init.W_expand # (d_model, d_model * expand_scale)
        self.B_expand = init.B_expand.reshape(1, 1, -1)

        self.W_shrink = init.W_shrink # (d_model * expand_scale,  d_model)
        self.B_shrink = init.B_shrink.reshape(1, 1, -1)

        self.activate = init.activate_function
        self.activate_backward = init.activate_function_backward

    def forward_train(self, X):
        # LN
        self.xnorm, self.layer_norm_cache = layer_norm(X) 
        
        # affine: xnorm = xhat * gamma + beta 
        self.xhat = (self.xnorm * self.gamma) + self.beta       # (B, T, d_model)

        # expand
        self.pre = self.xhat @ self.W_expand + self.B_expand    # (B, T, d_model * expand_scale)

        # activation
        self.act, self.activate_cache = self.activate(self.pre) # (B, T, d_model * expand_scale)

        # shrink 
        self.output = self.act @ self.W_shrink + self.B_shrink  # (B, T, d_model)
        return self.output
    
    d_B_shrink_list : list[npt.NDArray[np.float64]] = list()
    d_W_shrink_list : list[npt.NDArray[np.float64]] = list()

    d_B_expand_list : list[npt.NDArray[np.float64]] = list()
    d_W_expand_list : list[npt.NDArray[np.float64]] = list()

    d_beta_list : list[npt.NDArray[np.float64]] = list()
    d_gamma_list : list[npt.NDArray[np.float64]] = list()

    def backward(self, gradientOfOutput):
        d_B_shrink = gradientOfOutput.sum(axis=(0, 1), keepdims=True)
        d_W_shrink = self.act.transpose(0, 2, 1) @ gradientOfOutput

        self.d_B_shrink_list.append(d_B_shrink)
        self.d_W_shrink_list.append(d_W_shrink)

        d_act = gradientOfOutput @ self.W_shrink.transpose(0, 2, 1)

        d_pre = self.activate_backward(d_act, self.activate_cache)

        d_B_expand = d_pre.sum(axis=(0, 1), keepdims=True)
        d_W_expand = self.xhat.transpose(0, 2, 1) @ d_pre 

        self.d_B_expand_list.append(d_B_expand)
        self.d_W_expand_list.append(d_W_expand)

        d_xhat = d_pre @ self.W_expand.transpose(0, 2, 1)

        d_beta = d_xhat.sum(axis=(0, 1), keepdims=True)
        d_gamma = (d_xhat * self.xnorm).sum(axis=(0, 1), keepdims=True)

        self.d_beta_list.append(d_beta)
        self.d_gamma_list.append(d_gamma)

        d_xnorm = d_xhat * self.gamma

        dx = layer_norm_backward(d_xnorm, self.layer_norm_cache)
        return dx
