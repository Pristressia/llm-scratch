import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Layer_norm_cache:
    outputFromLN: npt.NDArray[np.float64]
    epsilon: float
    X: npt.NDArray[np.float64]
    inverse_varient: npt.NDArray[np.float64]
    d_model: int

def layer_norm(
        X: npt.NDArray[np.float64], 
        epsilon: float = 1e-5
        ) -> tuple[npt.NDArray, Layer_norm_cache]:
    
    mean = np.mean(X, axis=-1, keepdims=True)
    XminusMean = X - mean
    std = (XminusMean ** 2).mean(axis=-1, keepdims =True)
    invert_varient = 1 / np.sqrt(std + epsilon)

    d_model = X.shape[-1]

    output = XminusMean * invert_varient

    cache = Layer_norm_cache(
        outputFromLN = output,
        epsilon = epsilon,
        X = X,
        inverse_varient = invert_varient,
        d_model = d_model
    )

    return output, cache

def layer_norm_backward(
        gradientOfOutput: npt.NDArray[np.float64], 
        cache: Layer_norm_cache
        ):
    outputFromLN = cache.outputFromLN
    d_model = cache.d_model
    inverse_varient = cache.inverse_varient

    sum_gradient_over_d_model = np.sum(gradientOfOutput, axis=-1, keepdims=True)
    sum_gradient_times_output_over_d_model = np.sum(gradientOfOutput * outputFromLN, axis = -1, keepdims = True)

    gradient_of_layerNorm = (inverse_varient / d_model) * (
        d_model * gradientOfOutput 
        - sum_gradient_over_d_model 
        - outputFromLN * sum_gradient_times_output_over_d_model
        )

    return gradient_of_layerNorm