import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Relu_cache:
    mask: npt.NDArray[np.bool_] # true where X > 0


def Relu(X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], Relu_cache]:
    mask = X > 0
    return np.maximum(X, 0), Relu_cache(mask = mask);


def Relu_backward(gradient_of_output: npt.NDArray[np.float64], cache: Relu_cache) -> npt.NDArray[np.float64]:
    return gradient_of_output * cache.mask.astype(np.float64);