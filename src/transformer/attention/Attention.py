import numpy as np
import numpy.typing as npt
from dataclasses import dataclass 
from llm_operator import softmax, Softmax_cache, layer_norm, Layer_norm_cache, softmaxBackward, layer_norm_backward
from transformer.transformerCore.TransformerCore import TransformerCore
@dataclass
class Attention_cache:
    K: npt.NDArray[np.float64]
    Q: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]

@dataclass
class Attention_init:
    gamma: npt.NDArray[np.float64] # (1, 1, C) or (C, )
    beta: npt.NDArray[np.float64]  # (1, 1, C) or (C, )

    attentionHead: int             # H

    Wk: npt.NDArray[np.float64]    # (C, C) when C is hidden param or d_model
    Wq: npt.NDArray[np.float64]    # (C, C)
    Wv: npt.NDArray[np.float64]    # (C, C)

    merge_heads_bias: npt.NDArray[np.float64] # (1, 1, C) recommend

    @staticmethod
    def randomInitial(
        seeds: int = 1000, 
        B: int = 2, 
        T: int = 4, 
        d_model = 6
        ):
        
        C = d_model
        rng = np.random.default_rng(seeds)
        attention_init = Attention_init(
            gamma=  rng.random((1, 1, C), dtype = np.float64),
            beta= rng.random((1, 1, C), dtype=np.float64),
            attentionHead= 2,
            Wk=rng.random((1, C, C), np.float64),
            Wq = rng.random((1, C, C), np.float64),
            Wv = rng.random((1, C, C), np.float64),
            merge_heads_bias=rng.random((1, 1, C), np.float64),

        )
        return attention_init

class Attention(TransformerCore):

    K: npt.NDArray[np.float64]
    Q: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]
    H: int # number of attention head

    K_split: npt.NDArray[np.float64]
    Q_split: npt.NDArray[np.float64]
    V_split: npt.NDArray[np.float64]

    xhat: npt.NDArray[np.float64]
    xnorm: npt.NDArray[np.float64]
    Layer_norm_cache: Layer_norm_cache
    
    scores: npt.NDArray[np.float64]
    attentions: npt.NDArray[np.float64]
    output: npt.NDArray[np.float64]
    merge_output: npt.NDArray[np.float64]

    softmax_cache: Softmax_cache

    def __init__(
            self, 
            initial: Attention_init,
            name = "Attention1"
            ):
        
        self.Name = name

        self.Wk = initial.Wk.reshape(1, initial.Wk.shape[-2], initial.Wk.shape[-1])
        self.Wq = initial.Wq.reshape(1, initial.Wq.shape[-2], initial.Wq.shape[-1])
        self.Wv = initial.Wv.reshape(1, initial.Wv.shape[-2], initial.Wv.shape[-1])

        self.merge_heads_bias = initial.merge_heads_bias.reshape(1, 1, -1)

        self.gamma = initial.gamma.reshape(1, 1, -1)
        self.beta = initial.beta.reshape(1, 1, -1)

        if ( initial.Wk.shape[-2] != initial.Wk.shape[-1]):
            raise Exception(f"[Attention class] : shape of Wk is not allow {initial.Wk.shape}")
        
        if ( initial.Wq.shape[-2] != initial.Wq.shape[-1]):
            raise Exception(f"[Attention class] : shape of Wq is not allow {initial.Wq.shape}")
        
        if ( initial.Wv.shape[-2] != initial.Wv.shape[-1]): 
            raise Exception(f"[Attention class] : shape of Wv is not allow {initial.Wv.shape}")
        
        #d_model
        self.C = self.Wk.shape[-1]
        self.H = initial.attentionHead
        self.dHead = self.C // self.H

        self.d_merge_heads_bias_list: list[npt.NDArray[np.float64]] = list()
        self.d_Wk_list : list[npt.NDArray[np.float64]] = list()
        self.d_Wq_list : list[npt.NDArray[np.float64]] = list()
        self.d_Wv_list : list[npt.NDArray[np.float64]] = list()

        self.d_beta_list : list[npt.NDArray[np.float64]] = list()
        self.d_gamma_list : list[npt.NDArray[np.float64]] = list()

    @staticmethod
    def merge_heads(X:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        B, H, T, dHead = X.shape
        return X.transpose(0, 2, 1, 3).reshape(B, T, H * dHead)
    
    @staticmethod
    def split_heads(X:npt.NDArray[np.float64], attentionHead: int = 1) -> npt.NDArray[np.float64]:
        B, T, C = X.shape
        dHead = C // attentionHead
        X = X.reshape(B, T, attentionHead, dHead)
        return X.transpose(0, 2, 1, 3)

    def forward_train(self, X: npt.NDArray):
        """if this is the first transformer X must be layer norm before pass into this step"""

        self.xnorm, self.layer_norm_cache = layer_norm(X, epsilon=1e-5)
        self.xhat = self.xnorm * self.gamma + self.beta

        self.K = self.xhat @ self.Wk
        self.Q = self.xhat @ self.Wq
        self.V = self.xhat @ self.Wv

        self.K_split = self.split_heads(self.K, attentionHead=self.H)
        self.Q_split = self.split_heads(self.Q, attentionHead=self.H)
        self.V_split = self.split_heads(self.V, attentionHead=self.H)

        self.scores = self.Q_split @ self.K_split.transpose(0, 1, 3, 2) / np.sqrt(self.dHead)
        self.attentions, self.softmax_cache = softmax(self.scores, axis=-1)
        self.output = self.attentions @ self.V_split

        self.merge_output = self.merge_heads(self.output) + self.merge_heads_bias
        
        return self.merge_output

    def backward(self, gradientOfOutput: npt.NDArray[np.float64]):
        self.d_merge_heads_bias_list.append(gradientOfOutput.sum(axis=(0, 1), keepdims=True))

        d_merge_output = gradientOfOutput
        d_output = self.split_heads(d_merge_output, attentionHead=self.H)

        d_V_split = self.attentions.transpose(0, 1, 3, 2) @ d_output
        d_attentions = d_output @ self.V_split.transpose(0, 1, 3, 2)

        d_scores = softmaxBackward(d_attentions, self.softmax_cache)
        d_Q_split = d_scores @ self.K_split / np.sqrt(self.dHead)
        # d_K_split = (self.Q_split.transpose(0, 1, 3, 2) @ d_scores).transpose(0, 1, 3, 2) / np.sqrt(self.dHead)
        d_K_split = (d_scores.transpose(0, 1, 3, 2) @ self.Q_split) / np.sqrt(self.dHead) # chatgpt suggest this instead

        d_V = self.merge_heads(d_V_split)
        d_Q = self.merge_heads(d_Q_split)
        d_K = self.merge_heads(d_K_split)

        B, T, C = self.xhat.shape

        xhat_rs = self.xhat.reshape(B * T, C)
        
        d_K_rs = d_K.reshape(B * T, C)
        d_Q_rs = d_Q.reshape(B * T, C)
        d_V_rs = d_V.reshape(B * T, C)

        d_Wk = xhat_rs.T @ d_K_rs 
        d_Wq = xhat_rs.T @ d_Q_rs 
        d_Wv = xhat_rs.T @ d_V_rs 

        self.d_Wk_list.append(d_Wk)
        self.d_Wq_list.append(d_Wq)
        self.d_Wv_list.append(d_Wv)

        d_xhat_k = d_K @ self.Wk.transpose(0, 2, 1) 
        d_xhat_q = d_Q @ self.Wq.transpose(0, 2, 1) 
        d_xhat_v = d_V @ self.Wv.transpose(0, 2, 1) 

        d_xhat = d_xhat_k + d_xhat_q + d_xhat_v

        d_gamma = (d_xhat* self.xnorm).sum(axis=(0, 1), keepdims=True)
        d_beta = d_xhat.sum(axis=(0, 1), keepdims=True)

        self.d_gamma_list.append(d_gamma)
        self.d_beta_list.append(d_beta)

        d_xnorm = d_xhat * self.gamma

        d_x = layer_norm_backward(d_xnorm, self.layer_norm_cache)

        return d_x








