import numpy as np
import numpy.typing as npt

def softmax(
        X:npt.NDArray[np.float64], 
        axis: int = -1, 
        keepdims = True
        ) -> npt.NDArray[np.float64] :
    
    dX = X - np.max(X, axis = axis, keepdims = True)
    eX = np.exp(dX)
    return eX / np.sum(eX, axis = axis, keepdims = keepdims)


def softmaxBackward(
        P: npt.NDArray[np.float64], 
        gradientOfOutput: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]: 

    PtimesGrad = P * gradientOfOutput
    sum_PtimesGrad = np.sum(PtimesGrad, axis = -1, keepdims=True)
    return P * (gradientOfOutput - sum_PtimesGrad)