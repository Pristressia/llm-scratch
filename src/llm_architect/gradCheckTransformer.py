import numpy as np
import numpy.typing as npt

def gradCheckTransformer(forward, backward, eps=1e-6) -> None:
    rng = np.random.default_rng(1000)

    B, T, C = 2, 4, 6

    X = -rng.random((B, T, C), dtype=np.float64)
    G = -rng.random((B, T, C), dtype=np.float64) # upstream gradient

    # forward
    Y = forward(X)

    # analytical gradient
    dX = backward(G)

    # numerical gradient
    dX_num = np.zeros_like(X)
    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old = X[idx]

        X[idx] = old + eps
        Yp = forward(X)
        Lp = np.sum(Yp * G)

        X[idx] = old - eps
        Ym = forward(X)
        Lm = np.sum(Ym * G)

        X[idx] = old
        dX_num[idx] = (Lp - Lm) / (2 * eps)

        it.iternext()

    # compare

    max_abs = np.max(np.abs(dX - dX_num))
    rel = max_abs / (np.max(np.abs(dX) + np.abs(dX_num)) + 1e-22)


    print("max_abs_diff:", max_abs)
    print("relative_diff:", rel)