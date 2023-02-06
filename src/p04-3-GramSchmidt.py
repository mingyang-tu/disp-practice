import numpy as np
from utils.gram_schmidt import gram_schmidt


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    row_i, col_i = np.mgrid[0: 13, 0: 5].astype(np.float64)
    mat_A = row_i ** col_i
    gs = gram_schmidt(mat_A)
    print("Original:")
    print(mat_A)
    print("\nOrthonormal Vector Set:")
    print(gs)
    # print(gs.T.dot(gs))
