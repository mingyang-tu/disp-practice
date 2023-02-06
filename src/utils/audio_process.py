import numpy as np
from numpy.typing import NDArray
import wave


def read_wave(filename: str) -> tuple[wave._wave_params, NDArray[np.float64]]:
    with wave.open(filename, "rb") as f:
        params = f.getparams()
        str_data = f.readframes(params.nframes)

    wave_data = np.frombuffer(str_data, dtype=np.int16).astype(np.float64)
    wave_data = wave_data / max(abs(wave_data))
    wave_data = np.reshape(wave_data, [params.nframes, params.nchannels])

    return params, wave_data
