import numpy as np
from numpy.typing import NDArray


def projection(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return (np.dot(a, b) / np.dot(b, b)) * b


def gram_schmidt(basis: NDArray[np.float64]) -> NDArray[np.float64]:
    ortho = np.array(basis, dtype=np.float64)
    for i in range(ortho.shape[1]):
        for j in range(i):
            ortho[:, i] -= projection(ortho[:, i], ortho[:, j])
    return ortho / np.linalg.norm(ortho, axis=0)
