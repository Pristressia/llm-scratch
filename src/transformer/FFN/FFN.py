import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from llm_operator import layer_norm, layer_norm_backward, Layer_norm_cache

@dataclass
class FFN_init:

    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]

    W_expand: npt.NDArray[np.float64]
    B_expand: npt.NDArray[np.float64]

    W_shrink: npt.NDArray[np.float64]
    B_shrink: npt.NDArray[np.float64]

    activate_function: function
    activate_function_backward: function
    
class FFN:

    xnorm : npt.NDArray[np.float64]

    def __init__(self, init: FFN_init):
        self.gamma = init.gamma
        self.beta = init.beta

        self.W_expand = init.W_expand
        self.B_expand = init.B_expand

        self.W_shrink = init.W_shrink
        self.B_shrink = init.B_shrink

        self.activate

    def forward(self, X):
        self.xnorm, Layer_norm_cache = layer_norm(X)
        self.xhat = (self.xnorm @ self.gamma) + self.beta

        self.pre = self.xhat @ self.W_expand + self.B_expand
        self.act, 
