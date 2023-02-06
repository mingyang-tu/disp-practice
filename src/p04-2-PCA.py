import numpy as np
import matplotlib.pyplot as plt
from utils.pca import PCA


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    # example 1
    data_x = np.array([
        [2, -1,  3], [-1, 3, 5], [0,  2, 4],
        [4, -2, -1], [1,  0, 4], [-2, 5, 5]
    ], dtype=np.float64).T

    pca = PCA(data_x)
    print("Eigenvalue:")
    print(pca.eigval)
    print("\nEigenvector:")
    print(pca.eigvec)

    dim = 2
    data_x_new = pca.inverse_transform(pca.transform(dim)).astype(np.float64)
    print("\nOriginal data:")
    print(data_x)
    print(f"\nReduced to {dim} dimensions:")
    print(data_x_new)

    # example 2
    data_x = np.dot(np.random.rand(2, 2), np.random.randn(2, 200))

    pca = PCA(data_x)
    data_x_new = pca.inverse_transform(pca.transform(1))

    plt.figure("PCA Example")
    plt.scatter(data_x[0, :], data_x[1, :], alpha=0.2)
    plt.scatter(data_x_new[0, :], data_x_new[1, :], alpha=0.2)
    for v in pca.eigvec.T:
        plt.arrow(pca.mean[0], pca.mean[1], v[0], v[1],
                  color='r', width=0.01, head_width=0.05)
    plt.axis('equal')

    plt.show()
