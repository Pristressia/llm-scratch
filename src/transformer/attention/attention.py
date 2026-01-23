import numpy as np
import numpy.typing as npt
from dataclasses import dataclass 
from llm_operator import softmax, Softmax_cache, layer_norm, Layer_norm_cache

@dataclass
class Attention_cache:
    K: npt.NDArray[np.float64]
    Q: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]

@dataclass
class Attention_init:
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]

    attentionHead: int

    Wk: npt.NDArray[np.float64]
    Wq: npt.NDArray[np.float64]
    Wv: npt.NDArray[np.float64]

    merge_heads_bias: npt.NDArray[np.float64]
class Attention:

    K: npt.NDArray[np.float64]
    Q: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]

    xhat: npt.NDArray[np.float64]
    xnorm: npt.NDArray[np.float64]
    Layer_norm_cache: Layer_norm_cache
    
    scores: npt.NDArray[np.float64]
    attentions: npt.NDArray[np.float64]
    output: npt.NDArray[np.float64]
    merge_output: npt.NDArray[np.float64]

    def __init__(
            self, 
            initial: Attention_init,
            name = "Attention1"
            ):
        
        self.Name = name

        self.Wk = initial.Wk
        self.Wq = initial.Wq
        self.Wv = initial.Wv

        self.merge_heads_bias = initial.merge_heads_bias

        self.gamma = initial.gamma
        self.beta = initial.beta

        if (len(initial.Wk.shape) != 2 or initial.Wk.shape[0] != initial.Wk.shape[1]):
            raise Exception(f"[Attention class] : shape of Wk is not allow {initial.Wk.shape}")
        
        if (len(initial.Wq.shape) != 2 or initial.Wq.shape[0] != initial.Wq.shape[1]):
            raise Exception(f"[Attention class] : shape of Wq is not allow {initial.Wq.shape}")
        
        if (len(initial.Wv.shape) != 2 or initial.Wv.shape[0] != initial.Wv.shape[1]): 
            raise Exception(f"[Attention class] : shape of Wv is not allow {initial.Wv.shape}")
        
        #d_model
        self.C = initial.Wk.shape[0]
        self.dHead = self.C // initial.attentionHead

    @staticmethod
    def merge_heads(X:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        B, H, T, dHead = X.shape
        return X.transpose(0, 2, 1, 3).reshape(B, T, H, dHead)
    
    @staticmethod
    def split_heads(X:npt.NDArray[np.float64], attentionHead: int = 1) -> npt.NDArray[np.float64]:
        B, T, C = X.shape
        dHead = C // attentionHead
        X = X.reshape(B, T, attentionHead, dHead)
        return X.transpose(0, 2, 1, 3)

    def forward_train(self, X: npt.NDArray):
        """if this is the first transformer X must be layer norm before pass into this step"""

        self.xhat, self.layer_norm_cache = layer_norm(X, epsilon=1e-5)
        self.xnorm = self.xhat * self.gamma + self.beta

        self.K = self.xnorm @ self.Wk
        self.Q = self.xnorm @ self.Wq
        self.V = self.xnorm @ self.Wv

        K_split = self.split_heads(self.K, attentionHead=self.dHead)
        Q_split = self.split_heads(self.Q, attentionHead=self.dHead)
        V_split = self.split_heads(self.V, attentionHead=self.dHead)

        self.scores = Q_split @ K_split.transpose() / np.sqrt(self.dHead)
        self.attentions, softmax_cache = softmax(self.scores, axis=-1)
        self.output = self.attentions @ V_split

        self.merge_output = self.merge_heads(self.output) + self.merge_heads_bias


    def backward(self):
        






