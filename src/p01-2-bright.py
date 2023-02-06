import numpy as np
import matplotlib.pyplot as plt
from utils.bright import gamma, gamma_color


if __name__ == '__main__':
    root_path = "../images/"
    gray = plt.imread(root_path + "lena.bmp").astype(np.float64)
    lena = plt.imread(root_path + "lena512.bmp").astype(np.float64)

    plt.figure("Gamma")
    plt.imshow(gamma(gray, alpha=2), cmap="gray", vmin=0, vmax=255)

    plt.figure("Gamma Color")
    plt.imshow(gamma_color(lena, alpha=2).astype(np.uint8), vmin=0, vmax=255)

    plt.show()
