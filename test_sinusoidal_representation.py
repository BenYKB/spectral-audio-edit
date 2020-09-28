from unittest import TestCase
from sinusoidal_representation import *


class TestSinusoidalRepresentation(TestCase):
    def test_find_peaks(self):
        vals = np.array([1, 0, 0, 0, 0, 1, 2, 3, 4, 2, 3, 1.2, 1.1, 1, 0.9, 0.8, 0.5, 0])
        expected = np.array([0, 8])
        actual = find_peaks(vals)

        self.assertTrue(np.equal(actual, expected).all())

    def test_find_peaks_options(self):
        vals = np.array([1, 2, 3, 2, 4, 3, 2, 1])
        expected_1 = [4]
        expected_2 = [2, 4]
        expected_3 = [4]
        expected_4 = [4]

        actual_1 = find_peaks(vals, radius=2)
        actual_2 = find_peaks(vals, radius=1)
        actual_3 = find_peaks(vals, radius=1, minimum_height=3.5)
        actual_4 = find_peaks(vals, radius=1, maximum_number=1)

        self.assertTrue(np.equal(actual_1, expected_1).all())
        self.assertTrue(np.equal(actual_2, expected_2).all())
        self.assertTrue(np.equal(actual_3, expected_3).all())
        self.assertTrue(np.equal(actual_4, expected_4).all())

    def test_parabolic_peak(self):
        p1 = (0,0)
        p2 = (1,1)
        p3 = (2,0)

        expected = 1
        actual, _ = parabolic_peak(p1, p2, p3)
        self.assertTrue(np.isclose(expected, actual).all())

    def test_parabolic_peak_2(self):
        def y(x):
            return 9 * x ** 2 + 8 * x + 2

        expected = -8 / (2 * 9)
        actual, _ = parabolic_peak((1, y(1)), (2.3, y(2.3)), (-2, y(-2)))

        self.assertTrue(np.isclose(expected, actual).all())


