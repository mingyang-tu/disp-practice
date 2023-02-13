import numpy as np
import matplotlib.pyplot as plt
from utils.noise import add_gaussian_noise
from utils.filters import bilateral_filter_1d


if __name__ == '__main__':
    data_x = np.array([1] * 50 + [0] * 50, dtype=np.float64)
    data_y = add_gaussian_noise(data_x, sigma=1, amp=0.1)
    axis_n = np.arange(0, 100, dtype=np.int64)

    plt.figure("edge")
    plt.subplot(211)
    plt.plot(axis_n, data_x)
    plt.plot(axis_n, data_y)
    plt.subplot(212)
    plt.plot(axis_n, data_x)
    # plt.plot(n, y)
    plt.plot(axis_n, bilateral_filter_1d(data_y, 0.1, 5, 5))

    plt.show()
