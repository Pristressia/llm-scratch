import numpy as np
import numpy.typing as npt
from transformer.transformerCore.TransformerCore import TransformerCore


class DataPipe:

    name : str = "data"
    branchObject: TransformerCore

    def __init__(self, branchObject: TransformerCore ,name = "pipeNumber1"):
        self.name = name
        self.branchObject = branchObject
        pass

    def forward(self, X: npt.NDArray[np.float64], is_train: bool = True) :
        self.outputFromForward = self.branchObject.forward(X, is_train = is_train)
        self.outputFromPipe = X + self.outputFromForward
        return self.outputFromPipe

    def backward(self, gradientOfOutput):
        self.gradientOfX_sideway = self.branchObject.backward(gradientOfOutput)
        self.gradientOfInputPipe = gradientOfOutput + self.gradientOfX_sideway
        return self.gradientOfInputPipe
    
    

