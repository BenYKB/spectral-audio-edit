import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from phase_vocoder import principal_argument
from sinusoidal_representation import bin_to_angular_frequency, frequency_to_nearest_bin
from peak_picking import find_peaks, harmonic_comb_fundemental, whiten_spectrum
from envelope import get_time_series_envelope
from window import subset
import scipy.signal
import scipy.interpolate

class harmonic_profile:
    def __init__(self, harmonics):
        '''

        :param harmonics: list of (amplitude, angular frequency, phase) float triplet tuples
        '''

        harmonics.sort(key=lambda x: x[1])
        harmonics.sort(key=lambda x: x[0], reverse=True)

        max_amplitude = harmonics[0][0]
        fundamental_frequency = harmonics[0][1]
        fundamental_phase = harmonics[0][2]

        norm_harmonics = list(map(lambda x: (x[0] / max_amplitude,
                                             x[1] / fundamental_frequency,
                                             principal_argument(x[2] - fundamental_phase)),
                                  harmonics))
        self.harmonics = norm_harmonics

        # fundemental amplitude + fundemental frequency = 1, fundemental phase is 0
        # (f, a, p): (1,1,0), (2, 0.5, 0), (3, 0.1, -0.3), (0.5, 0.3, 0.1) <- subharmonic like oboe
        #


def extract_harmonics(signal, sample_rate, cutoff_freq=30, peak_radius=50, peak_number=7):
    f_sig = np.fft.rfft(signal)
    cutoff_bin = frequency_to_nearest_bin(cutoff_freq, signal.size, sample_freq=sample_rate)

    amp = np.absolute(f_sig)

    peaks = list(filter(lambda i: i > cutoff_bin,
                        find_peaks(amp, radius=peak_radius, maximum_number=peak_number)))

    return set(map(lambda i: (amp[i], bin_to_angular_frequency(i, signal.size), np.angle(f_sig[i])),
                   peaks))


def get_center_of_interest(signal):
    i = np.arange(signal.size)
    squared = signal ** 2
    return int(round(sum(i * squared) / sum(squared)))



if __name__ == '__main__':
    import os, re

    sample_size = 4096 * 2

    rate, signal_oboe = wavfile.read(
        r'C:\Users\elber\Documents\AudioRecordings\PhilharmoniaSamples\oboe\macro-output\oboe_A4_1_mezzo-forte_normal.wav')
    rate, signal_trumpet = wavfile.read(
        r'C:\Users\elber\Documents\AudioRecordings\PhilharmoniaSamples\trumpet\macro-output\trumpet_A4_025_forte_normal.wav')
    rate, signal_flute = wavfile.read(
        r'C:\Users\elber\Documents\AudioRecordings\PhilharmoniaSamples\flute\macro-output\flute_A4_15_mezzo-forte_normal.wav')

    instrument_signals = np.array([signal_oboe, signal_trumpet, signal_flute])

    # ft_oboe = np.abs(np.fft.rfft(signal_oboe))
    # whitened = whiten_spectrum(ft_oboe, rate)
    #
    # print(whitened.size)
    # print(ft_oboe.size)
    #
    # plt.plot(np.arange(whitened.size), np.log(whitened), label='whitened')
    # #plt.plot(np.arange(ft_oboe.size), np.log(ft_oboe), label='original')
    # plt.legend()
    # plt.show()


    ft_o = np.abs(np.fft.rfft(signal_oboe))
    eps = np.average(ft_o) / 2


    print(len(instrument_signals))
    print(f'sample rate is {rate}')
    # higher harmonics show many small peaks. Perhaps from vibrato or transients

    centers = np.array([get_center_of_interest(inst_sig) for inst_sig in instrument_signals])
    radius = sample_size // 2

    sound_samples = []
    hamming = scipy.signal.windows.hamming(sample_size)

    for i in range(len(instrument_signals)):
        sig = instrument_signals[i]
        c = centers[i]

        sound_samples.append(hamming * subset(sig, c - radius, c + radius))


    #f_0, harmonics = harmonic_comb_fundemental(np.abs(np.fft.rfft(sound_samples[0])))


    path = r'C:\Users\elber\Documents\AudioRecordings\PhilharmoniaSamples\oboe\macro-output'

    thirty_hz_cutoff = int(30.0 / 44100.0 * sample_size)
    five_khz_cutoff = int(5000.0 / 44100.0 * sample_size)
    pattern = re.compile("As3")
    illegal_patterns = re.compile("trill|phrase")

    data = []

    for filename in os.listdir(path):
        if pattern.search(filename) and not illegal_patterns.search(filename):
            print(f"read {filename}")
            rate, sig = wavfile.read(os.path.join(path, filename))
            center = get_center_of_interest(sig)
            sample = np.pad(hamming * subset(sig, center -radius, center + radius), (sample_size*2, sample_size))
            abs_ft = np.abs(np.fft.rfft(sample))
            eps = np.average(abs_ft)

            f_0, harmonics = harmonic_comb_fundemental(abs_ft, lowerbound_index=thirty_hz_cutoff, upperbound_index=five_khz_cutoff, interpolate=True)

            data.append((f_0 / sample_size * rate, harmonics, eps, filename))

            harmonics = harmonics / harmonics[0]
            fs = (f_0 * np.arange(1, harmonics.size + 1))[0:5]
            log_harmonics = np.log(harmonics[0:5] + eps*np.ones(5))

            plt.plot(fs, log_harmonics, marker='|')
            plt.hlines(np.log(eps), 0,400)
            #plt.show()

    plt.show()

    print(data)

    #we want to store (file_name, f_0 in hz, harmonic strengths (raw), avg_mag)
    '''we want to reject files labled trill or phrase'''



    #breakpoint()
    # for sample in sound_samples:
    #     #wavfile.write(r'C:\Users\elber\Documents\AudioRecordings' + f"\\instrument{n}" + ".wav", rate, sample)
    #     n += 1
    #     ft = np.abs(np.fft.rfft(sample))
    #     eps = np.log(np.average(ft))
    #     ft = np.log(ft)
    #     #plt.plot(np.arange(sample.size), sample)
    #     interp_sig = scipy.interpolate.interp1d(np.arange(ft.size), ft, kind='cubic')
    #     if n == 3:
    #         plt.hlines((eps),0,2000)
    #         plt.scatter(np.arange(ft.size), ft)
    #         plt.plot(np.linspace(0, ft.size-1, num=ft.size * 10), interp_sig(np.linspace(0, ft.size -1 , num=ft.size * 10)))
    #
    # plt.show()
    #
    #
    # #for i in np.arrange()
    #
    # windowed_samples = scipy.signal.windows.hamming(sample_size)
    #
    #
    # f_signal_oboe = np.fft.rfft(signal_oboe)
    #
    # amplitude_oboe = np.absolute(np.fft.rfft(signal_oboe))
    # amplitude_trumpet = np.absolute(np.fft.rfft(signal_trumpet))
    # amplitude_flute = np.absolute(np.fft.rfft(signal_flute))
    #
    # cutoff_freq = 30
    # cutoff_bin = frequency_to_nearest_bin(cutoff_freq, signal_oboe.size, sample_freq=rate)
    # oboe_peaks = list(filter(lambda i: i > cutoff_bin, find_peaks(amplitude_oboe, radius=50, maximum_number=7)))
    # oboe_harmonics = set(map(lambda i: (amplitude_oboe[i],
    #                                     bin_to_angular_frequency(i, signal_oboe.size),
    #                                     np.angle(f_signal_oboe[i])),
    #                          oboe_peaks))
    # print(oboe_harmonics)
    #
    # print(extract_harmonics(signal_oboe, rate))
    #
    #
    #
    # plt.scatter(oboe_peaks, amplitude_oboe[oboe_peaks])
    # plt.plot(np.arange(amplitude_oboe.size), amplitude_oboe)
    # plt.show()
    #
    # plt.clf()
    # plt.plot(np.arange(amplitude_trumpet.size), amplitude_trumpet)
    # plt.show()
    #
    # plt.clf()
    # plt.plot(np.arange(amplitude_flute.size), amplitude_flute)
    # plt.show()
