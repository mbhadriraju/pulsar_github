import psrchive
import numpy as np
from .preprocess import load_data

def noise(freq, time, max_data):
    r = np.random.uniform(0.9, 1.0, size=time)
    w = np.random.normal(0, 1.0, size=time)
    jumps = np.where(np.random.random(size=time) < 0.1, np.random.standard_t(df=1, size=time), 0.0)

    arr = np.zeros(time)
    arr[0] = w[0]
    
    sig = arr[0]
    for i in range(1, time):
        sig = (sig * r[i]) + (w[i] * np.sqrt(1 - r[i]**2)) + jumps[i]
        arr[i] = sig

    i_arr = np.arange(1, freq + 1)[:, np.newaxis]
    noise_arr = arr * (1 / (i_arr ** 2))

    red_noise_arr = noise_arr * (max_data * np.random.uniform(0.01, 6))

    white_noise_arr = np.random.normal(0, 1, size=(freq, time)) * (max_data * np.random.uniform(0.01, 6))

    return red_noise_arr + white_noise_arr


def generate_individual(template_path, n, output_path, i, path):
    signal, arch = load_data(template_path)

    exp = np.random.random() * 2
    indices = np.arange(1, signal.shape[0] + 1)[:, np.newaxis]
    signal = signal * (1 / indices ** exp)

    shift = np.random.randint(-signal.shape[0] // 2, signal.shape[0] // 2)
    signal = np.roll(signal, shift, axis=0)

    max_signal = np.max(signal) / 30

    denoise_signal = signal.copy()

    noise_arr = noise(signal.shape[0], signal.shape[1], max_signal)

    signal = signal + noise_arr

    return np.array([signal, denoise_signal])
