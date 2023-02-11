import numpy as np
import matplotlib.pyplot as plt
from utils.filters import lowpass_mask_img


if __name__ == '__main__':
    root_path = "../images/"
    lena = plt.imread(root_path + "lena.bmp").astype(np.float64)
    lena_fft = np.fft.fft2(lena)
    baboon = plt.imread(root_path + "Baboon.bmp").astype(np.float64)
    baboon_fft = np.fft.fft2(baboon)

    low = lowpass_mask_img(lena.shape, 30)
    high = 1. - low

    img_new = np.fft.ifft2(lena_fft * low + baboon_fft * high).real

    plt.figure("Low")
    plt.imshow(low, cmap="gray")
    plt.figure("High")
    plt.imshow(high, cmap="gray")
    plt.figure("Output Image")
    plt.imshow(img_new, cmap="gray", vmin=0, vmax=255)

    plt.show()
