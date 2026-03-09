from typing import Dict, List
import numpy as np
import numpy.typing as npt

class Embedding:

    embeddingDict: npt.NDArray[np.float64]
    d_model: int

    def __init__(self, d_model, embeddingDict):
        self.embeddingDict = embeddingDict
        self.d_model = d_model

    def embed(self, words: List[str]) :

        embededText = np.zeros((len(words), self.d_model), dtype=np.float64)
        nullEmbed = np.zeros((0, self.d_model), dtype=np.float64)

        for i in range(len(words)):
            isContent = any(words[i] == dictWord for dictWord in self.embeddingDict.keys())
            if isContent :
                embededText[i][:] = self.embeddingDict[words[i]]
            else :
                embededText[i][:] = nullEmbed

        return np.array(embededText)
    
    def deEmbed(self, embeddedText: npt.NDArray[np.float64]):
        
