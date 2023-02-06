import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    root_path = "../images/"
    lena = plt.imread(root_path + "lena.bmp").astype(np.float64)
    binary = np.round(lena / 255)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    plt.figure("Binary")
    plt.imshow(binary, cmap="gray")
    plt.figure("Erosion")
    plt.imshow(cv2.erode(binary, kernel, iterations=3), cmap="gray")
    plt.figure("Dilation")
    plt.imshow(cv2.dilate(binary, kernel, iterations=3), cmap="gray")
    plt.figure("Opening")
    plt.imshow(cv2.erode(
        cv2.dilate(binary, kernel, iterations=3),
        kernel,
        iterations=3
    ), cmap="gray")
    plt.figure("Closing")
    plt.imshow(cv2.dilate(
        cv2.erode(binary, kernel, iterations=3),
        kernel,
        iterations=3
    ), cmap="gray")

    plt.show()
