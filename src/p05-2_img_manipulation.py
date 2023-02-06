import numpy as np
import matplotlib.pyplot as plt
from utils.image_manipulation import resize_image, affine_transformation


if __name__ == '__main__':
    root_path = "../images/"
    lena = plt.imread(root_path + "lena.bmp").astype(np.float64)

    row, col = lena.shape

    plt.figure("Original")
    plt.imshow(lena, cmap="gray", vmin=0, vmax=255)
    plt.figure("Bilinear Interpolation")
    plt.imshow(
        resize_image(lena, (int(1.5 * row), int(1.6 * col))),
        cmap="gray", vmin=0, vmax=255
    )

    rotation = np.array([
        [np.cos(np.pi / 6), np.sin(np.pi / 6)],
        [-np.sin(np.pi / 6), np.cos(np.pi / 6)]
    ], dtype=np.float64)

    shearing = np.array([[1, 0], [0.3, 1]], dtype=np.float64)

    plt.figure("Rotation")
    plt.imshow(
        affine_transformation(lena, rotation),
        cmap="gray", vmin=0, vmax=255
    )
    plt.figure("Shearing")
    plt.imshow(
        affine_transformation(lena, shearing),
        cmap="gray", vmin=0, vmax=255
    )

    plt.show()
