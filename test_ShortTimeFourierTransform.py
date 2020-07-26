from unittest import TestCase
from short_time_fourier_transform import *
import numpy as np


class TestShortTimeFourierTransform(TestCase):
    def test_apply(self):

        self.fail()

    def test_apply_inverse(self):
        window_len = 4

        stft = ShortTimeFourierTransform(window_len)

        expected = np.arange(10, dtype=np.float32)
        actual = stft.apply_inverse(stft.apply(expected,window_len//2),window_len//2)[window_len//2:window_len//2+10]

        self.assertTrue((np.abs(expected - actual) < 0.0001).all())

    def test_cartesian_to_polar(self):
        self.fail()

    def test_polar_to_cartesian(self):
        self.fail()
