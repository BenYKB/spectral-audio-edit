import cmath
import math

class MorletWavelet:
    def __init__(self, f, n, cutoff_decays):
        self.frequency = f
        self.n = n
        self._k_complex = 2j * cmath.pi * self.frequency
        self._k_real = 2 * (cmath.pi * self.frequency / self.n) ** 2
        self.radius = math.sqrt(2 * cutoff_decays) * n / (2 * math.pi * f)

    def apply(self, t):
        return cmath.exp(self._k_complex * t - t ** 2 * self._k_real)
