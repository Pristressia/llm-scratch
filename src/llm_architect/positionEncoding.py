import numpy as np

def positionEncoding(T: int, d_model: int) :
    position_encodings = np.zeros((T, d_model))

    for pos in range(T):
        for i in range(d_model):
            if i % 2 == 0:
                position_encodings[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            else :
                position_encodings[pos, i] = np.cos(pos / (10000 ** (2 * i / d_model)))
    
    return position_encodings