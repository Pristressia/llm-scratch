import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Softmax_cache:
    propability : npt.NDArray[np.float64]


def softmax(
        X:npt.NDArray[np.float64], 
        axis: int = -1, 
        keepdims = True
        ) -> tuple[npt.NDArray[np.float64], Softmax_cache] :
    
    dX = X - np.max(X, axis = axis, keepdims = True)
    eX = np.exp(dX)
    propability = eX / np.sum(eX, axis = axis, keepdims = keepdims)
    return propability, Softmax_cache(propability = propability)


def softmaxBackward(
        gradientOfOutput: npt.NDArray[np.float64],
        cache: Softmax_cache
        ) -> npt.NDArray[np.float64]: 

    P = cache.propability
    PtimesGrad = P * gradientOfOutput
    sum_PtimesGrad = np.sum(PtimesGrad, axis = -1, keepdims=True)
    return P * (gradientOfOutput - sum_PtimesGrad)


def test(x):
    return x * 100