import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Layer_norm_cache:
    outputFromLN: npt.NDArray[np.float64] # normalized output
    epsilon: float
    X: npt.NDArray[np.float64]
    inverse_std: npt.NDArray[np.float64]  # inverse standard deviation = 1 / \sqrt(var + \epsilon)
    d_model: int

def layer_norm(
        X: npt.NDArray[np.float64], 
        epsilon: float = 1e-5
        ) -> tuple[npt.NDArray[np.float64], Layer_norm_cache]:
    
    mean = np.mean(X, axis=-1, keepdims=True)
    XminusMean = X - mean
    var = (XminusMean ** 2).mean(axis=-1, keepdims =True)
    invert_varient = 1 / np.sqrt(var + epsilon)

    d_model = X.shape[-1]

    output = XminusMean * invert_varient

    cache = Layer_norm_cache(
        outputFromLN = output,
        epsilon = epsilon,
        X = X,
        inverse_std = invert_varient,
        d_model = d_model
    )

    return output, cache

def layer_norm_backward(
        gradientOfOutput: npt.NDArray[np.float64], 
        cache: Layer_norm_cache
        ):
    outputFromLN = cache.outputFromLN
    d_model = float(cache.d_model)
    inverse_std = cache.inverse_std

    # (B,T,1)
    sum_gradient_over_d_model = np.sum(gradientOfOutput, axis=-1, keepdims=True)
    sum_gradient_times_output_over_d_model = np.sum(gradientOfOutput * outputFromLN, axis = -1, keepdims = True)

    gradient_of_layerNorm = (inverse_std / d_model) * (
        d_model * gradientOfOutput 
        - sum_gradient_over_d_model 
        - outputFromLN * sum_gradient_times_output_over_d_model
        )

    return gradient_of_layerNorm