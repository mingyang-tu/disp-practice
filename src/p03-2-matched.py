import numpy as np
import matplotlib.pyplot as plt
from utils.filters import match_filter


if __name__ == '__main__':
    data_x = np.concatenate([
        np.zeros(10), np.ones(10),
        np.zeros(10), np.arange(-5, 6) / 5,
        np.zeros(10), np.arange(5, -6, -1) / 5,
        np.zeros(10), np.sin(np.arange(10) / np.pi * 2),
        np.zeros(10)
    ], dtype=np.float64)

    axis_n = np.arange(0, data_x.shape[0], dtype=np.int64)

    plt.figure("match filter")
    plt.subplot(211)
    plt.stem(axis_n, data_x)
    plt.subplot(212)
    plt.stem(axis_n, match_filter(data_x, np.arange(-5, 6) / 10))

    plt.show()
