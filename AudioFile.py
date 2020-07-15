import numpy as np


class AudioFile:
    def __init__(self, signal, sample_rate):
        """signal has more than 1 entry, sample_rate in Hz"""
        self._s = signal
        self._rate = sample_rate

    def sample_rate(self):
        return self._rate

    def duration(self):
        """duration in seconds"""
        return self.length() / self._rate

    def length(self):
        """length of signal array"""
        return np.size(self._s)

    def subset(self, start_index, end_index):
        """Subset by array index, zero pads if index out of bounds.
        One index must be in bounds
        start_index < end_index
        end_index not included
         """
        l = start_index
        r = end_index
        l_pad = 0
        r_pad = 0

        if start_index < 0:
            l = 0
            l_pad = -1 * start_index

        if end_index > self.length():
            r = self.length()
            r_pad = end_index - self.length()

        return np.pad(self._s[l:r], (l_pad, r_pad))

    def subsets(self, length, hop, offset=0):
        """subset length, step size, starting subset offset.
        All ints. length must be divisible by 2
        Returns (center index of subsets), (subsets).
        Hop must be less than length"""

        n = self.length() // hop
        centers = np.arange(n+1) * hop + offset
        r = length // 2

        if length % 2 == 0:
            subsets = [self.subset(c - r, c + r) for c in centers]
        else:
            subsets = [self.subset(c - r, c + r + 1) for c in centers]

        return centers, subsets



