from unittest import TestCase
from phase_vocoder import *
from short_time_fourier_transform import *
import numpy as np
import matplotlib.pyplot as plt


class TestTimeFrequencyEffects(TestCase):
    def test_time_stretch_phases(self):
        time = np.arange(16*50, dtype=np.float32)
        sine_wave = np.pad(np.sin(time * 3.3 / 32 * 2 * np.pi)*np.hanning(time.size), (10, 10))

        window_len = 32
        ana_hop = 32 // 4
        syn_hop = int(ana_hop * 2)

        plt.plot(np.arange(sine_wave.size), sine_wave)
        plt.show()

        stft = ShortTimeFourierTransform(window_len)

        a = stft.apply(sine_wave, ana_hop)
        plt.clf()
        plt.imshow(np.absolute(a))
        plt.show()

        mag, phases = cartesian_to_polar(a)
        plt.clf()
        plt.plot(np.arange(phases[3].size), phases[3], label='initial')

        u_phases = time_strech_phases(phases, window_len, ana_hop, syn_hop)

        plt.plot(np.arange(u_phases[3].size), u_phases[3], label='mod')
        plt.legend()
        plt.show()

        mod_a = polar_to_cartesian(mag, u_phases)

        b = stft.apply_inverse(mod_a, syn_hop)

        c = stft.apply(b, ana_hop)

        plt.clf()
        plt.imshow(np.absolute(c))
        plt.show()

        plt.clf()
        plt.plot(np.arange(b.size), b)
        plt.show()

        self.fail()

    def test_principal_argument(self):
        a = principal_argument(0.2)
        a_e = 0.2

        b = principal_argument(-0.2)
        b_e = -0.2

        c = principal_argument(np.pi)
        c_e = np.pi

        d = principal_argument(-2.6 * np.pi)
        d_e = -0.6 * np.pi

        e = principal_argument(2.6 * np.pi)
        e_e = 0.6 * np.pi

        f = principal_argument(-8.2 * np.pi)
        f_e = -0.2 * np.pi

        g = principal_argument(9.2 * np.pi)
        g_e = -0.8 * np.pi

        self.assertTrue(abs(a - a_e) < 0.0001)
        self.assertTrue(abs(b - b_e) < 0.0001)
        self.assertTrue(abs(c - c_e) < 0.0001 or abs(c + c_e) < 0.0001)
        self.assertTrue(abs(d - d_e) < 0.0001)
        self.assertTrue(abs(e - e_e) < 0.0001)
        self.assertTrue(abs(f - f_e) < 0.0001)
        self.assertTrue(abs(g - g_e) < 0.0001)
