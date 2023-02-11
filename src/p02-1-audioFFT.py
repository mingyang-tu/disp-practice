import numpy as np
import matplotlib.pyplot as plt
from utils.audio_process import read_wave


if __name__ == '__main__':
    root_path = "../images/"
    params, wave_data = read_wave(root_path + "twinkle.wav")

    fs, nframes = params.framerate, params.nframes
    n0 = int(np.ceil(nframes / 2))

    fft_data = np.fft.fftshift(np.abs(np.fft.fft(wave_data[:, 0])) / fs)

    freq = np.concatenate([range(n0 - nframes, 0), range(0, n0)]) * fs / nframes

    plt.figure("FFT")
    plt.plot(freq, fft_data)
    plt.xlim(-1000, 1000)

    plt.show()
