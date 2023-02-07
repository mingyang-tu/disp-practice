import numpy as np
from numpy.typing import NDArray
from typing import Union


class PCA:
    def __init__(self, data: NDArray[np.float64]) -> None:
        self.dim, self.num_sample = data.shape
        self.mean = np.mean(data, axis=1)

        self.X = data - self.mean.reshape(-1, 1)

        eig = np.linalg.eig(np.dot(self.X, self.X.T))
        order = sorted(range(len(eig[0])), key=lambda x: eig[0][x], reverse=True)

        self.eigval = eig[0][order]
        self.eigvec = eig[1][:, order]
        # eigvec = [u1 u2 ... um]
        # eigvec[:, i] is the eigenvector corresponding to eigval[i]

    def get_P(self, dim: int = 0) -> NDArray[Union[np.float64, np.complex128]]:
        if dim > 0:
            return self.eigvec[:, :dim]
        return self.eigvec

    def transform(self, dim: int = 0) -> NDArray[Union[np.float64, np.complex128]]:
        P = self.get_P(dim).T
        return np.dot(P, self.X)

    def inverse_transform(self, X_pca: NDArray[Union[np.float64, np.complex128]]) -> NDArray[Union[np.float64, np.complex128]]:
        dim = X_pca.shape[0]
        P = self.get_P(dim)
        return np.dot(P, X_pca) + self.mean.reshape(-1, 1)
