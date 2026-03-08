import numpy as np
import numpy.typing as npt
from typing import Callable, Any
from dataclasses import dataclass
from llm_operator import layer_norm, layer_norm_backward, Layer_norm_cache, Relu, Relu_backward
from transformer.transformerCore.TransformerCore import TransformerCore
import os
from config import CHECKPOINT_DIR

from helper import getRootPath

ActForward = Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], Any]]
ActBackward = Callable[[npt.NDArray[np.float64], Any], npt.NDArray[np.float64]]
0
class FFN_init:

    gamma: npt.NDArray[np.float64]    # (1, 1, d_model) or (d_model, )
    beta: npt.NDArray[np.float64]     # (1, 1, d_model) or (d_model, )

    W_expand: npt.NDArray[np.float64] # (1, d_model, d_model * expand_scale) or (d_model, d_model * expand_scale)
    B_expand: npt.NDArray[np.float64] # (1, 1, d_model * expand_scale) or (d_model * expand_scale, )

    W_shrink: npt.NDArray[np.float64] # (1, d_model * expand_scale, d_model) or (d_model * expand_scale, d_model)
    B_shrink: npt.NDArray[np.float64] # (1, 1 * d_model) or (d_model, )

    activate_function: ActForward
    activate_function_backward: ActBackward

    def __init__ (
            self, 
            gamma: npt.NDArray[np.float64], 
            beta: npt.NDArray[np.float64], 
            W_expand: npt.NDArray[np.float64], 
            B_expand: npt.NDArray[np.float64], 
            W_shrink: npt.NDArray[np.float64], 
            B_shrink: npt.NDArray[np.float64], 
            activate_function, 
            activate_function_backward
            ) :
        self.gamma = gamma
        self.beta = beta
        self.W_expand = W_expand
        self.B_expand = B_expand
        self.W_shrink = W_shrink
        self.B_shrink = B_shrink

        self.activate_function = activate_function
        self.activate_function_backward = activate_function_backward
        pass

    @staticmethod
    def randomInitial(
        d_model: int, 
        expand_scale: int, 
        seed: int=2000, 
        activate_function = Relu, 
        activate_function_backword = Relu_backward
        ) :
        rng = np.random.default_rng(seed)
        return FFN_init(
            gamma = rng.random((1, 1, d_model), dtype = np.float64),
            beta = rng.random((1, 1, d_model), dtype = np.float64),

            W_expand = rng.random((1, d_model, d_model * expand_scale), dtype = np.float64),
            B_expand = rng.random((1, 1, d_model * expand_scale), dtype = np.float64),
            
            W_shrink = rng.random((1, d_model * expand_scale, d_model), dtype = np.float64),
            B_shrink = rng.random((1, d_model, 1), dtype = np.float64),

            activate_function = activate_function,
            activate_function_backward = activate_function_backword
        )
    
    
class FFN(TransformerCore):

    xnorm : npt.NDArray[np.float64]

    def __init__(self, init: FFN_init, name: str ="FFN1"):
        self.gamma = init.gamma.reshape(1, 1, -1)
        self.beta = init.beta.reshape(1, 1, -1)

        self.W_expand = init.W_expand # (d_model, d_model * expand_scale)
        self.B_expand = init.B_expand.reshape(1, 1, -1)

        self.W_shrink = init.W_shrink # (d_model * expand_scale,  d_model)
        self.B_shrink = init.B_shrink.reshape(1, 1, -1)

        self.activate = init.activate_function
        self.activate_backward = init.activate_function_backward

        self.d_B_shrink_list : list[npt.NDArray[np.float64]] = list()
        self.d_W_shrink_list : list[npt.NDArray[np.float64]] = list()

        self.d_B_expand_list : list[npt.NDArray[np.float64]] = list()
        self.d_W_expand_list : list[npt.NDArray[np.float64]] = list()
        
        self.d_beta_list : list[npt.NDArray[np.float64]] = list()
        self.d_gamma_list : list[npt.NDArray[np.float64]] = list()

        self.Name = name

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
    
    def commit_model(self):
        self.gamma_infer = self.gamma.reshape(1, self.gamma.shape[-1])
        self.beta_infer = self.beta.reshape(1, self.beta.shape[-1])

        C = self.W_expand.shape[-2]
        E = self.W_expand.shape[-1]

        self.W_expand_infer = self.W_expand.reshape(C, E)
        self.B_expand_infer = self.B_expand.reshape(1, E)

        self.W_shrink_infer = self.W_shrink.reshape(E, C)
        self.B_shrink_infer = self.B_shrink.reshape(1, C)
    
    def forward_infer(self, X):
        xnorm, _ = layer_norm(X)
        xhat = (xnorm * self.gamma_infer) + self.beta_infer

        pre = xhat @ self.W_expand_infer + self.B_expand_infer
        act, _ = self.activate(pre)

        output = act @ self.W_shrink_infer + self.B_shrink_infer
        return output
    
    def forward(
            self, 
            X: npt.NDArray[np.float64], 
            is_train: bool = True
            ):
            
            
        if (is_train): 
            return self.forward_train(X)
        return self.forward_infer(X)

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
    

    def save_checkpoint(self):
        rootPath = getRootPath();
        filePath = os.path.join(rootPath, CHECKPOINT_DIR, self.Name + ".ffn.npz")  
        
        np.savez(
            file = filePath,

            gamma = self.gamma,
            beta = self.beta,

            W_expand = self.W_expand,
            B_expand = self.B_expand,
            
            W_shrink = self.W_shrink,
            B_shrink = self.B_shrink
        )

        print(f"FFN {self.Name} has been save successfully @ {filePath}")


    @staticmethod
    def load_checkpoint(name: str, activation_function, activation_function_backward) :
        rootPath = getRootPath();
        filePath = os.path.join(rootPath, CHECKPOINT_DIR, name + ".ffn.npz")  
        data = np.load(file = filePath);

        ffn_init = FFN_init(
            gamma = data["gamma"], 
            beta = data["beta"], 
            activate_function=activation_function, 
            activate_function_backward = activation_function_backward,
            W_expand=data["W_expand"],
            B_expand=data["B_expand"],
            W_shrink=data["W_shrink"],
            B_shrink=data["B_shrink"]
        )

        return FFN(ffn_init, name=name)


        
