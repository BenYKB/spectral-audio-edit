from window import *
import numpy as np


def cartesian_to_polar(stft_cartesian):
    """
    Converts a short time fourier transform in cartesian coordinates to polar
    :param stft: 2-d numpy array short time fourier transform in cartesian coordinates
    :return: 2-d numpy array (magnitude), 2-d numpy array (radians between (-pi,pi])
    """

    return np.absolute(stft_cartesian), np.angle(stft_cartesian)


def polar_to_cartesian(magnitudes, angles):
    """
    Inverts cartesian_to_polar. Takes magnitudes and phases of stft and converts to cartesian coordinates
    :param magnitudes: 2-d numpy array of magnitudes
    :param angles: 2-d numpy array of phases of equal dimension to magnitudes
    :return: 2-d numpy of cartesian coordinates
    """

    return magnitudes * (np.complex(np.cos(angles), np.sin(angles)))


class ShortTimeFourierTransform:
    def __init__(self, window_length):
        self._w = Window(window_length)

    def apply(self, signal, analysis_hop_length):
        """
        Apply short time fourier transform
        :param signal: 1-d numpy array time series
        :return: 2-d numpy array short time fourier transform
        """
        frames = center(
                    self._w.hann(
                        self._w.subsets(signal, analysis_hop_length)))
        output = np.fft.rfft(frames)
        return output

    def apply_inverse(self, stft_cartesian, synthesis_hop_length):
        """
        Invert the short time fourier transform
        :param stft: 2-d numpy array short time fourier transform in cartesian coordinates
        :return: 1-d numpy array time series
        """
        number_of_frames, _fft_size = stft_cartesian.shape

        output_length = synthesis_hop_length * (number_of_frames - 1) + self.window_length()

        out = np.zeros(output_length, dtype=np.float32)
        norm = np.zeros(output_length, dtype=np.float32)
        w_norm = self._w.hann(np.ones(self.window_length())) ** 2

        for i_frame in np.arange(number_of_frames):
            i_out = i_frame * synthesis_hop_length
            out[i_out:i_out+self.window_length()] += self._w.hann(
                                                        invert_center(
                                                            np.fft.irfft(stft_cartesian[i_frame],
                                                                         self.window_length())))
            norm[i_out:i_out+self.window_length()] += w_norm

        norm[0] = 1
        return out / norm

    def window_length(self):
        return self._w.length()






