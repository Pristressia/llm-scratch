import numpy as np
import numpy.typing as npt

class TransformerCore:

    name : str = "core1"

    def __init__(self, name: str = "core1"):
        self.name = name


    def forward(self, X: npt.NDArray[np.float64]) :
        return X
    
    def backward(self, gradientOfOutput: npt.NDArray[np.float64]):
        return gradientOfOutput
    
    def forward_train(self, X: npt.NDArray[np.float64]):
        return X