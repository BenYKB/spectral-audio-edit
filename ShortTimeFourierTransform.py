from Window import Window
import numpy as np


class ShortTimeFourierTransform:
    def __init__(self, sample_rate, window_length, hop_length):
        self._rate = sample_rate
        self._w = Window(window_length)
        self._hop_length = hop_length

    def apply(self, signal):
        """
        Apply short time fourier transform
        :param signal: 1-d numpy array time series
        :return: 2-d numpy array short time fourier transform
        """
        frames = self._w.center(
                    self._w.hann(
                        self._w.subsets(signal, self._hop_length)))
        output = np.fft.rfft(frames)
        return output

    def apply_inverse(self, stft_cartesian):
        """
        Invert the short time fourier transform
        :param stft: 2-d numpy array short time fourier transform in cartesian coordinates
        :return: 1-d numpy array time series
        """
        pass


    def convert_to_polar(self, stft_cartesian):
        """
        Converts a short time fourier transform in cartesian coordinates to polar
        :param stft: 2-d numpy array short time fourier transform in cartesian coordinates
        :return: 2-d numpy array (magnitude), 2-d numpy array (radians between (-pi,pi])
        """

        return np.absolute(stft_cartesian), np.angle(stft_cartesian)

    def sample_rate(self):
        return self._rate

    def hop_length(self):
        return self._hop_length

    def window_length(self):
        return self._w.length()
