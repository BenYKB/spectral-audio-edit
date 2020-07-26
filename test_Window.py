from unittest import TestCase
from window import *
import numpy as np


class TestWindow(TestCase):
    def test_hann(self):
        w = Window(4)

        a = w.hann(np.array([1, 1, 1, 1]))
        self.assertTrue(abs(a[1] - a[3]) < 0.001)
        self.assertTrue(abs(a[0]) < 0.001 )

    def test_center(self):
        a = np.array([1,2,3,4])
        actual = center(a)
        expected = np.array([3,4,1,2])
        self.assertTrue((expected==actual).all())

    def test_subset(self):
        a = np.arange(11)
        expected = np.array([0, 1])
        actual = subset(a, 0, 2)

        self.assertTrue((expected==actual).all())

    def test_subsets(self):
        w = Window(4)

        a = np.arange(5)
        expected = np.array([np.array([0, 0, 0, 1]), np.array([0, 1, 2, 3]), np.array([2, 3, 4, 0])])

        self.assertTrue((w.subsets(a, 2) == expected).all())

    def test_invert_center(self):
        a = np.array([1,2,3,4])
        b = invert_center(center(a))

        self.assertTrue((a==b).all())