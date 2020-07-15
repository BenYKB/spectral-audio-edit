import numpy as np


class TimeFrequencyEffects:
    def __init__(self, sample_rate, analysis_hop, analysis_window):
        self.rate = sample_rate
        self.syn_hop = analysis_hop
        self.syn_win = analysis_window

    def time_stretch_phases(self, phases, synthesis_hop):
        del_p_analysis = phases[1:] - phases[:-1]
        bin_freqs = 2*np.pi * np.arange(self.syn_win//2) / self.syn_win  # considering rfft so we divide by 2
        additional_phase_shift = self.principal_argument(del_p_analysis - bin_freqs*synthesis_hop)
        true_freqs = additional_phase_shift / synthesis_hop + bin_freqs
        l, _ = phases.shape
        out = phases.copy()

        # syn phase at t = syn phase t-1 + syn hop * true freq
        for i in range(1,l):
            out[i] = out[i-1] + synthesis_hop * true_freqs[i-1]  # true freqs should be 1 shorter than phases so this is really t_f[i]

        return out


    def principal_argument(self, theta):
        return theta - np.floor(theta/np.pi) * np.pi
