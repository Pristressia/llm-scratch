from typing import Dict, List
import numpy as np
import numpy.typing as npt

class Embedding:

    embeddingDict: List[npt.NDArray[np.float64]]
    d_model: int

    def __init__(self, d_model: int, embeddingDict: List[npt.NDArray[np.float64]]):
        self.embeddingDict = embeddingDict
        self.d_model = d_model

    def embed(self, wordIndex: int) :

        if (wordIndex > len(self.embeddingDict) - 1) :
            raise Exception("word index > size of embeddict")
        
        return self.embeddingDict[wordIndex]
    

    def backward(self, )
