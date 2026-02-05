import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

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
