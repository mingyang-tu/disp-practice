import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import nrmse, psnr, ssim
from utils.noise import add_gaussian_image


if __name__ == '__main__':
    root_path = "../images/"
    gray = plt.imread(root_path + "lena.bmp").astype(np.float64)

    noisy = add_gaussian_image(gray, 10)

    print(f"NRMSE = {nrmse(gray, noisy)}")
    print(f"PSNR = {psnr(gray, noisy)}")
    print(f"SSIM = {ssim(gray, noisy)}")

    plt.figure("Original")
    plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
    plt.figure("Noisy")
    plt.imshow(noisy, cmap="gray", vmin=0, vmax=255)

    plt.show()
