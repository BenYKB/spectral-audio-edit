import numpy as np


class Window:
    def __init__(self, length):
        self._length = length
        self._hann = self._get_hann()

    def _get_hann(self):
        times = np.arange(self._length)
        return np.sin(times * np.pi / (self._length - 1)) ** 2

    def hann(self, signal):
        """Returns signal windowed by Hann window"""
        return signal * self._hann

    def length(self):
        return self._length

    def center(self, signal):
        """
        Circularly shifts signal so the middle of the signal occurs at the t=0
        :param signal: numpy array containing at least one element
        :return: numpy array centered via circular shifting
        """
        r = -1 * signal.size // 2 if signal.size % 2 == 0 else -1 * signal.size // 2 + 1
        return np.roll(signal, r)

    def subset(self, signal, start_index, end_index):
        """
        A subset of the signal with zero padding if necessary
        :param signal: numpy array of values
        :param start_index: start index of the subset. One of start_index and end_index must be in bounds for signal.
        :param end_index: end index of the subset, non-inclusive.
        :return: subset of signal, numpy array
        """

        l = start_index
        r = end_index
        l_pad = 0
        r_pad = 0

        if start_index < 0:
            l = 0
            l_pad = -1 * start_index

        if end_index > signal.size:
            r = signal.length()
            r_pad = end_index - signal.length()

        return np.pad(signal[l:r], (l_pad, r_pad))

    def subsets(self, signal, hop_size, offset=0):
        """
        Returns all subsets of signal of width of this window.
        :param signal: numpy array to be split
        :param hop_size: elements between each subset center, less than this window length
        :param offset: index at which the first subset is centered
        :return: signal split into subsets
        """

        n = signal.size // hop_size
        centers = np.arange(n+1) * hop_size + offset
        r = self._length // 2

        if self._length % 2 == 0:
            subsets = [self.subset(signal, c - r, c + r) for c in centers]
        else:
            subsets = [self.subset(signal, c - r, c + r + 1) for c in centers]

        return centers, subsets
