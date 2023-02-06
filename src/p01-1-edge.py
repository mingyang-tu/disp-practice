import numpy as np
import matplotlib.pyplot as plt
from utils.edge import difference, sobel, laplacian


if __name__ == '__main__':
    simple = np.array([
        [11,  10,  10,  10,  12,  11,  10,   9,  10],
        [10,  10,  11,  10,  10,  10,  10,  11,   9],
        [10,  10,   9, 150, 150, 150,  10,  10,  10],
        [10,  10, 160, 160, 155, 160, 158,  10,  11],
        [10,  10, 158, 160, 161, 161, 160, 150,  10],
        [10, 155, 160, 163, 164, 165, 160, 151,  10],
        [10, 148, 160, 160, 162, 160, 155,  10,  12],
        [8,   10, 140, 150, 152, 150,  10,  11,  10],
        [9,   12,  10,  10,  10,  10,   9,  10,  10]
    ], dtype=np.float64)

    plt.figure("difference")
    plt.imshow(abs(difference(simple, 0)), cmap="gray", vmin=0, vmax=255)

    plt.figure("Sobel")
    plt.imshow(abs(sobel(simple, 0)), cmap="gray", vmin=0, vmax=255)

    plt.figure("Laplacian")
    plt.imshow(abs(laplacian(simple)), cmap="gray", vmin=0, vmax=255)

    plt.show()
