import numpy as np
from scipy.io import wavfile
from short_time_fourier_transform import *
import matplotlib.pyplot as plt


def principal_argument(theta):
    return (theta + np.pi) - (np.floor((theta + np.pi) / np.pi / 2) * np.pi * 2) - np.pi


def time_strech_phases(phases, window_length, analysis_hop, synthesis_hop):
    bin_freqs = 2 * np.pi * np.arange(window_length // 2 + 1) / window_length  # considering rfft so we divide by 2
    additional_phase_shift = principal_argument(phases[1:] - phases[:-1] - bin_freqs * analysis_hop)
    true_freqs = additional_phase_shift / analysis_hop + bin_freqs
    l, _ = phases.shape
    out = phases.copy()

    # syn phase at t = syn phase t-1 + syn hop * true freq
    # TODO: implement as a complex exponential multiple of the cartsian stft
    for i in range(1, l):
        out[i] = out[i - 1] + synthesis_hop * true_freqs[
            i - 1]  # true freqs should be 1 shorter than phases so this is really t_f[i]

    return out


class PhaseVocoder:
    def __init__(self, window_length=2048):
        self.w_len = 2048
        self._stft = ShortTimeFourierTransform(self.w_len)

    def time_stretch(self, signal, stretch_factor, analysis_hop=None):
        """
        Stretches signal in time by stretch factor without changing pitch

        :param signal: 1-d numpy array of audio signal
        :param stretch_factor: positive float (bigger than one increases signal length)
        :param analysis_hop: Specific analysis hop can be specified (defaults to w_len //4 if not specified)
        :return: 1-d numpy array of stretched signal
        """

        if not analysis_hop:
            analysis_hop = self._stft.window_length() // 4

        synthesis_hop = int(analysis_hop * stretch_factor)
        analysis_transform = self._stft.apply(signal, analysis_hop)

        magnitudes, phases = cartesian_to_polar(analysis_transform)
        new_phases = time_strech_phases(phases, self.window_length(), analysis_hop, synthesis_hop)

        synthesis_transfrom = polar_to_cartesian(magnitudes, new_phases)

        return self._stft.apply_inverse(synthesis_transfrom, synthesis_hop)

    def pitch_shift(self, signal, shift_factor):
        pass

    def window_length(self):
        return self._stft.window_length()


if __name__ == "__main__":
    rate_1, sig_1 = wavfile.read(r'C:\Users\elber\Documents\AudioRecordings\bumble.wav')

    assert (sig_1.dtype == np.float32)

    window_len = 2048
    analysis_hop = window_len // 6
    synthesis_hop = int(analysis_hop / 1.5)

    stft = ShortTimeFourierTransform(window_len)
    analysis_transform = stft.apply(sig_1, analysis_hop)

    plt.imshow(np.absolute(analysis_transform))
    plt.show()

    mags, phases = cartesian_to_polar(analysis_transform)
    new_phases = time_strech_phases(phases, window_len, analysis_hop, synthesis_hop)
    new_transform = polar_to_cartesian(mags, new_phases)

    new_signal = stft.apply_inverse(new_transform, synthesis_hop)

    plt.clf()
    plt.imshow(np.absolute(stft.apply(new_signal, analysis_hop)))
    plt.show()

    wavfile.write(r'C:\Users\elber\Documents\AudioRecordings\PV_Output_1_5_v2.wav', rate_1, new_signal)
