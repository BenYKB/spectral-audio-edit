import numpy as np
from scipy.io import wavfile


def principal_argument(theta):
    return (theta + np.pi) - (np.floor((theta + np.pi) / np.pi / 2) * np.pi * 2) - np.pi

def time_strech_phases(phases, window_length, analysis_hop, synthesis_hop):
    bin_freqs = 2 * np.pi * np.arange(window_length // 2 + 1) / window_length # considering rfft so we divide by 2
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
