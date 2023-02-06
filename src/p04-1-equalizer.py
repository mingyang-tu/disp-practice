import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.noise import add_gaussian_image
from utils.filters import blur_filter_img, wiener_filter_img


if __name__ == '__main__':
    root_path = "../images/"
    lena = plt.imread(root_path + "lena.bmp").astype(np.float64)
    filt = blur_filter_img(L=10, sigma=0.1)
    blurred = signal.convolve2d(lena, filt, boundary='symm', mode='same')
    blurred_gaussian = add_gaussian_image(blurred, 5)

    # plt.figure("Filter")
    # plt.imshow(filt, cmap="gray")
    plt.figure("Original")
    plt.imshow(lena, cmap="gray", vmin=0, vmax=255)
    plt.figure("Blurred + Noise")
    plt.imshow(blurred_gaussian, cmap="gray", vmin=0, vmax=255)
    plt.figure("Recover")
    plt.imshow(wiener_filter_img(blurred_gaussian, filt, 0.01), cmap="gray", vmin=0, vmax=255)

    plt.show()
