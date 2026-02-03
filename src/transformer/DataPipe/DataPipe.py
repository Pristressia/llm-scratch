import numpy as np


class DataPipe:

    name : str = "data"

    def __init__(self, name = "pipeNumber1"):
        self.name = name
        pass

    def forward(self, X, forwardFunction) :
        self.outputFromForward = forwardFunction(X)
        self.outputFromPipe = X + self.outputFromForward
        return self.outputFromPipe

    def backward(self, gradientOfOutput, backwardFunction):
        self.gradientOfX_sideway = backwardFunction(gradientOfOutput)
        self.gradientOfInputPipe = gradientOfOutput + self.gradientOfX_sideway
        return self.gradientOfInputPipe
    
    

