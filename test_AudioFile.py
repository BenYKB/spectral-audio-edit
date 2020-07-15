from unittest import TestCase
from AudioFile import AudioFile
import numpy as np

class TestAudioFile(TestCase):

    def testSubset(self):
        a = AudioFile(np.arange(11), 1)
        expected = np.array([0,1])
        actual = a.subset(0,2)
        self.assertTrue((expected==actual).all())

    def testSubsets(self):
        a = AudioFile(np.arange(5), 1)
        actual_c, actual_s = a.subsets(4, 2)
        expected_c = np.array([0,2,4])
        expected_s = np.array([np.array([0,0,0,1]), np.array([0,1,2,3]), np.array([2,3,4,0])])
        print(actual_c)
        print(actual_s)

        self.assertTrue((actual_c==expected_c).all())
        self.assertTrue((actual_s==expected_s).all())


    def testDuration(self):
        pass



