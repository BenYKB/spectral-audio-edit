import numpy as np

import matplotlib.pyplot as plt

from FunctionGenerator import FunctionGenerator

# def convolve(signal, kernel):
#     f_signal = np.fft.fft(signal)
#     f_kernel = np.fft.fft(kernel)
#
#     return np.fft.ifft(f_convolved)

def to_length(signal, length):
    pad_length = length - signal.size
    return np.pad(signal, (0, pad_length)) if pad_length > 0 else signal


if __name__ == '__main__':
    func_gen = FunctionGenerator(10)
    sig = func_gen.square_time_series(1.5, 1, 0.55, 0.5)
    print("signal")
    print(sig)
    kernel = np.array([0,1,1,0])
    print("kernel")
    print(kernel)
    print("kernal at length")
    kernel = to_length(kernel, sig.size)
    print(kernel)

    f_sig = np.fft.fft(sig)
    f_kernel = np.fft.fft(kernel)

    f_result = f_sig * f_kernel

    convolution_result = np.real(np.fft.ifft(f_result))

    print("Convolution")
    print(convolution_result)

    plt.plot(convolution_result, np.arange(convolution_result.size))
    plt.show()
