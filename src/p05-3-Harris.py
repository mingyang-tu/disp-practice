import numpy as np
import matplotlib.pyplot as plt
from utils.harris import harris_corner
from utils.filters import blur_filter_img


if __name__ == '__main__':
    root_path = "../images/"
    left01 = plt.imread(root_path + "left01.jpg").astype(np.float64)

    window = blur_filter_img(3, 0.3)

    corners, mat_R = harris_corner(left01, window, 0.04, 1e6)
    corners_y, corners_x = zip(*corners)

    plt.figure("Corners")
    plt.imshow(left01, cmap="gray", vmin=0, vmax=255)
    plt.plot(corners_x, corners_y, 'o', color='red', markersize=2)
    plt.figure("Harris")
    plt.imshow(mat_R, cmap="jet")
    plt.colorbar()
    plt.show()
