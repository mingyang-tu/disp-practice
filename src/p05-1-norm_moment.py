import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import norm, central_moment_2d


if __name__ == '__main__':
    row_i, col_i = np.mgrid[-100: 101, -150: 151].astype(np.float64)
    ellipse = (((row_i / 90) ** 2 + (col_i / 140) ** 2) <= 1).astype(np.float64)

    plt.figure("ellipse")
    plt.imshow(ellipse, cmap="gray")

    print(f"L_0 = {norm(ellipse, 0)}")
    print(f"L_1 = {norm(ellipse, 1)}")
    print(f"L_2 = {norm(ellipse, 2)}")
    print(f"L_inf = {norm(ellipse, np.inf)}\n")

    print(f"m_2,0 = {central_moment_2d(ellipse, 2, 0)}")
    print(f"m_1,1 = {central_moment_2d(ellipse, 1, 1)}")
    print(f"m_0,2 = {central_moment_2d(ellipse, 0, 2)}")

    plt.show()
