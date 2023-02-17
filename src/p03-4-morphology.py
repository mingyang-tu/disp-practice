import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from utils.morphology import dilation, erosion


if __name__ == '__main__':
    root_path = "../images/"
    lena = plt.imread(root_path + "lena.bmp").astype(np.float64)

    binary = np.round(lena / 255).astype(np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    iters = 3

    start1 = time.time()
    img_erode = erosion(binary, kernel, iterations=iters)
    end1 = time.time()

    start2 = time.time()
    cv_erode = cv2.erode(binary, kernel, iterations=iters)
    end2 = time.time()

    print(f"Erode: {np.max(cv_erode - img_erode)}")
    print(f"numpy: {end1 - start1}, opencv: {end2 - start2}")

    start1 = time.time()
    img_dilate = dilation(binary, kernel, iterations=iters)
    end1 = time.time()

    start2 = time.time()
    cv_dilate = cv2.dilate(binary, kernel, iterations=iters)
    end2 = time.time()

    print(f"Dilate: {np.max(cv_dilate - img_dilate)}")
    print(f"numpy: {end1 - start1}, opencv: {end2 - start2}")

    img_open = erosion(img_dilate, kernel, iterations=iters)
    print(f"Open: {np.max(cv2.erode(cv_dilate, kernel, iterations=iters) - img_open)}")

    img_close = dilation(img_erode, kernel, iterations=iters)
    print(f"Close: {np.max(cv2.dilate(cv_erode, kernel, iterations=iters) - img_close)}")

    plt.figure("Binary")
    plt.imshow(binary, cmap="gray")

    plt.figure("Erosion")
    plt.imshow(img_erode, cmap="gray")

    plt.figure("Dilation")
    plt.imshow(img_dilate, cmap="gray")

    plt.figure("Opening")
    plt.imshow(img_open, cmap="gray")

    plt.figure("Closing")
    plt.imshow(img_close, cmap="gray")

    plt.show()
