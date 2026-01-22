import numpy as np
import numpy.typing as npt

class Attention:

    def __init__(
            self, 
            initWk: npt.NDArray[np.float64], 
            initWq: npt.NDArray[np.float64], 
            initWv: npt.NDArray[np.float64], 
            attentionHead:int = 1, 
            name = "Attention1"
            ):
        
        self.Name = name
        self.Wk = initWk
        self.Wq = initWq
        self.Wv = initWv

        if (len(initWk.shape) != 2 or initWk.shape[0] != initWk.shape[1]):
            raise Exception(f"[Attention class] : shape of Wk is not allow {initWk.shape}")
        
        if (len(initWq.shape) != 2 or initWq.shape[0] != initWq.shape[1]):
            raise Exception(f"[Attention class] : shape of Wq is not allow {initWq.shape}")
        
        if (len(initWv.shape) != 2 or initWv.shape[0] != initWv.shape[1]): 
            raise Exception(f"[Attention class] : shape of Wv is not allow {initWv.shape}")
        
        #d_model
        self.C = initWk.shape[0]
        self.dHead = self.C // attentionHead

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

    def forward(self, X: npt.NDArray):
