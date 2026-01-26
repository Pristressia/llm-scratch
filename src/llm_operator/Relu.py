import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Relu_cache:
    X: npt.NDArray[np.float64]


def Relu(X: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return np.maximum(X, 0);


def Relu_backward(gradient_of_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (X > 0).astype(np.float64);