import numpy as np
import matplotlib.pyplot as plt
import short_time_fourier_transform as stft
from functools import reduce
from scipy.io import wavfile
import window as win
from phase_vocoder import principal_argument
import scipy.signal
from envelope import get_time_series_envelope
from peak_picking import *

def estimate_angular_freqs_amplitudes(peridiogram_frame, peak_indicies, fft_size):
    out_frequency = []
    out_amplitude = []

    for i in peak_indicies:
        if i == 0:
            out_frequency.append(bin_to_angular_frequency(i, fft_size))
            out_amplitude.append(np.sqrt(peridiogram_frame[i]))
        elif i == peridiogram_frame.size - 1:
            out_frequency.append(bin_to_angular_frequency(i, fft_size))
            out_amplitude.append(np.sqrt(peridiogram_frame[i]))
        else:
            points = list(zip(range(i-1, i+2), 0.5*np.log(peridiogram_frame[i-1:i+2])))
            #print(points)
            bin, log_amp = parabolic_peak(*points)
            out_frequency.append(bin_to_angular_frequency(bin, fft_size))
            out_amplitude.append(np.exp(log_amp))

    return np.array(out_frequency), np.array(out_amplitude)


def bin_to_angular_frequency(bin_number, fft_length, sample_freq=None):
    w = 2 * np.pi * bin_number / fft_length
    return w if not sample_freq else w * sample_freq


def bin_to_frequency(bin_number, fft_length, sample_freq=None):
    w = bin_number / fft_length
    return w if not sample_freq else w * sample_freq


def frequency_to_nearest_bin(freq, fft_length, sample_freq=None):
    return np.round(freq * fft_length) if not sample_freq else np.round(freq * fft_length / sample_freq)


def put_in_set_or_make_new(dictionary, key, value):
    if key in dictionary:
        dictionary[key].add(value)
    else:
        dictionary[key] = {value}


def sinusoids_at_time(tracks, t):
    pass


class SinusoidalRepresentation:
    '''
    Algorithm based on "Speech analysis/Synthesis based on a sinusoidal representation" by R. McAulay and T. Quatieri
    '''
    def __init__(self, signal, window_size=2048, sample_rate=44100):
        self.tracks = self.from_signal(signal)
        self.window_size = window_size
        self.rate = sample_rate

        # self.peak_finder = win.Window(5)

    def from_signal(self, signal):
        return []

    def from_tracks(self, tracks):
        pass


class Track:
    def __init__(self, sinusoids: list):
        """
        Contiguous track of sinusoids
        :param sinusoids: list of sinusoids. Must contain at least one element
        """
        self.sinusoids = sinusoids

    def start_frame(self):
        return self.sinusoids[0].t

    def last_frame(self):
        return self.sinusoids[-1].t

    def track_length(self):
        return self.last_frame() - self.start_frame()

    def is_defined_at_time(self, t):
        return self.start_frame() <= t <= self.last_frame()

    def get_nearest_sinusoid(self, t):
        return reduce(lambda a, b: a if a is not None and abs(a.t - t) < abs(b.t - t) else b, self.sinusoids, None)

    def get_max_amplitude(self):
        return reduce(lambda a, b: a if a > b.a else b.a, self.sinusoids, 0.0)

    def get_average_angular_frequency(self):
        return sum(map(lambda x: x.w, self.sinusoids)) / len(self.sinusoids)

    def apply_phase_frequency_refinement(self):
        num_sins = len(self.sinusoids)
        if num_sins < 2:
            return

        original_freqs = list(map(lambda x: x.w, self.sinusoids))
        phases = list(map(lambda x: x.p, self.sinusoids))
        times = list(map(lambda x: x.t, self.sinusoids))

        forward_time_step = times[1] - times[0]

        forward_difference = principal_argument(
            phases[1] - phases[0] - original_freqs[0] * forward_time_step) / forward_time_step

        self.sinusoids[0].w = forward_difference + self.sinusoids[0].w

        for i in range(1, num_sins - 1):
            forward_time_step = times[i + 1] - times[i]
            forward_difference = principal_argument(
                phases[i + 1] - phases[i] - original_freqs[i] * forward_time_step) / forward_time_step
            backward_time_step = times[i] - times[i - 1]
            backward_difference = principal_argument(
                phases[i] - phases[i - 1] - original_freqs[i] * backward_time_step) / backward_time_step

            self.sinusoids[i].w = self.sinusoids[i].w + 0.5 * (forward_difference + backward_difference)

        backward_time_step = times[num_sins - 1] - times[num_sins - 2]
        backward_difference = principal_argument(phases[num_sins - 1] - phases[num_sins - 2] - original_freqs[
            num_sins - 1] * backward_time_step) / backward_time_step
        self.sinusoids[num_sins - 1].w = backward_difference + self.sinusoids[num_sins - 1].w
        # TODO: Consider bin shift error...

    def taper(self, taper_length, custom_opposite_length=None):
        first_taper = taper_length
        last_taper = taper_length if not custom_opposite_length else custom_opposite_length

        first_sinusoid = track.sinusoids[0]
        last_sinusoid = track.sinusoids[-1]

        if first_sinusoid.a != 0.0:
            track.sinusoids.insert(0,
                                   Sinusoid(0.0,
                                            first_sinusoid.w,
                                            first_sinusoid.p - first_sinusoid.w * first_taper,
                                            first_sinusoid.t - first_taper))
        if last_sinusoid.a != 0.0:
            track.sinusoids.append(Sinusoid(0.0,
                                            last_sinusoid.w,
                                            last_sinusoid.p + last_sinusoid.w * last_taper,
                                            last_sinusoid.t + last_taper))

    def to_time_series(self):
        out = []
        for n in range(len(self.sinusoids) - 1):
            out.extend(
                self.sinusoids[n].interpolate(self.sinusoids[n + 1])
            )

        return np.array(out)


class Sinusoid:
    def __init__(self, amplitude, frequency, phase, time):
        """
        Sinusoidal estimate
        :param amplitude: amplitude of the sinusoid
        :param frequency: angular frequency (omega)
        :param phase: phase of sinusoid
        :param time: time samples
        """
        self.a = amplitude
        self.w = frequency
        self.p = phase
        self.t = time

    def is_close(self, other):
        if isinstance(other, Sinusoid):
            return np.isclose(np.array([self.a, self.w, self.p, self.t]),
                              np.array([other.a, other.w, other.p, other.t])).all()
        else:
            return False

    def deep_copy(self):
        return Sinusoid(self.a, self.w, self.p, self.t)

    def interpolate(self, next_sinusoid):
        times = np.arange(next_sinusoid.t - self.t)
        delta_t = times.size

        amplitudes = self.a + (next_sinusoid.a - self.a) * (times / delta_t)

        a = next_sinusoid.p - self.p - self.w * delta_t
        b = next_sinusoid.w - self.w
        m = np.round((-1 * a + b * delta_t * 0.5) / (2 * np.pi))
        alpha = 3 / delta_t ** 2 * (a + 2 * np.pi * m) + -1 / delta_t * b
        beta = -2 / delta_t ** 3 * (a + 2 * np.pi * m) + 1 / delta_t ** 2 * b
        phases = self.p + self.w * times + alpha * times ** 2 + beta * times ** 3

        return amplitudes * np.cos(phases)

    # TODO: Test methods for working


if __name__ == '__main__':
    # TODO: active windowing sizing

    rate, signal = wavfile.read(r'C:\Users\elber\Documents\AudioRecordings\input_signal_32.wav')
    signal = np.pad(signal, (2048 * 2, 2048 * 2))

    ##
    ## Analysis Section
    ##

    envelope = get_time_series_envelope(signal, rate)

    note_activation = 0.13

    # plt.plot(np.arange(signal.size), signal)
    # plt.plot(np.arange(envelope.size), envelope)
    # plt.axhline(note_activation)
    # plt.title('Original signal')
    # plt.show()
    # plt.clf()

    print(f'max in input: {np.amax(signal)}')
    assert (signal.dtype == np.float32)

    window_length = 512
    analysis_step = 512
    tracking_radius = 1 * bin_to_angular_frequency(1, window_length)  # bins * (freq step)
    max_peaks = 70
    minimum_height = 0.00000
    analysis_window = win.Windows.HAMMING
    analysis_stft = stft.ShortTimeFourierTransform(window_length)
    stft_frames = analysis_stft.apply(signal, analysis_step, window_type=analysis_window)
    peridiogram_frames = np.real(stft_frames) ** 2 + np.imag(stft_frames) ** 2
    reflection_amplitude_correction = 2

    # plt.imshow(np.transpose(np.log(peridiogram_frames)), origin='lower')
    # plt.show()
    # plt.clf()

    num_frames, _ = stft_frames.shape

    tracks = []
    active_tracks = []

    wt = []

    for frame_step in range(num_frames):
        current_peridiogram = peridiogram_frames[frame_step]
        current_stft = stft_frames[frame_step]

        peaks_index = find_peaks(current_peridiogram, maximum_number=max_peaks, minimum_height=minimum_height)
        peak_frequencies, peak_amplitudes = estimate_angular_freqs_amplitudes(current_peridiogram, peaks_index, window_length)
        peak_amplitudes = peak_amplitudes * reflection_amplitude_correction
        # peak_amplitudes = np.sqrt(np.take(current_peridiogram, peaks_index)) * reflection_amplitude_correction
        # peak_frequencies = bin_to_angular_frequency(peaks_index, window_length)


        # print(f"z phase {np.take(current_stft, peaks_index)}")
        # print(f"a phase {np.angle(np.take(current_stft, peaks_index))}")

        peak_phases = np.angle(np.take(current_stft, peaks_index))

        fundemental = histogram_fundemental(peaks_index)
        if fundemental:
            wt.append((fundemental, frame_step * analysis_step))

        # previous_f = 0
        # if True:
        #     plt.plot(np.arange(peridiogram_frames[frame_step].size), np.log(peridiogram_frames[frame_step]))
        #     plt.vlines(peaks_index, 0,1)
        #     fundemental = historgram_fundemental(peaks_index)
        #     plt.vlines(fundemental, -10,1)
        #     plt.title(f'Peridiogram at frame: {frame_step} with matches at {peaks_index}]')
        #     if fundemental is not None and fundemental != previous_f:
        #         plt.show()
        #     previous_f = fundemental
        #     plt.clf()

        frame_sinusoids = set()
        tentative_tracks = dict()

        for n in range(peaks_index.size):
            s = Sinusoid(peak_amplitudes[n],
                         peak_frequencies[n],
                         peak_phases[n],
                         frame_step * analysis_step)

            frame_sinusoids.add(s)

            for track in active_tracks:
                distance = abs(track.sinusoids[-1].w - s.w)
                if distance <= tracking_radius:
                    put_in_set_or_make_new(tentative_tracks, track, (s, distance))

        next_active_tracks = []

        for sinusoid in frame_sinusoids:
            s_matches = []
            for t in tentative_tracks.keys():
                for (s, d) in tentative_tracks[t]:
                    if sinusoid.is_close(s):
                        s_matches.append((t, d))

            if len(s_matches) != 0:
                s_matches.sort(key=lambda x: x[0].sinusoids[-1].w)
                s_matches.sort(key=lambda x: x[1])
                t_match = s_matches[0][0]
                t_match.sinusoids.append(sinusoid)

                next_active_tracks.append(t_match)
                tentative_tracks.pop(t_match)

                for (t, d) in s_matches[1:]:
                    updated_set = set(filter(lambda x: not x[0].is_close(sinusoid), tentative_tracks[t]))
                    if len(updated_set) == 0:
                        tentative_tracks.pop(t)
                    else:
                        tentative_tracks[t] = updated_set

            else:
                next_active_tracks.append(Track([sinusoid]))

        dead_tracks = list(filter(lambda x: x not in next_active_tracks, active_tracks))
        tracks.extend(dead_tracks)
        active_tracks = next_active_tracks

    tracks.extend(active_tracks)

    for track in tracks:
        #track.apply_phase_frequency_refinement()
        # TODO: Test phase freq vs. peak estimation
        track.taper(analysis_step)
        t = list(map(lambda x: x.t, track.sinusoids))
        w = list(map(lambda x: x.w * 44100/np.pi/2, track.sinusoids))
        #plt.plot(t, w)

    #plt.show()

    ##
    ## Editing Section
    ##

    upper_w = 5000 / rate * 2 * np.pi
    lower_w = 30 / rate * 2 * np.pi

    sinusoid_frames = []
    fundemental_sinusoids = []

    for t in np.arange(signal.size, step=analysis_step):
        current_tracks = filter(lambda x: x is not None and x.is_defined_at_time(t), tracks)
        current_sinusoids = list(map(lambda x: x.get_nearest_sinusoid(t), current_tracks))
        current_sinusoids.sort(key=lambda x: (x.w, -x.a))

        candidate_s = list(filter(lambda x: upper_w > x.w > lower_w, current_sinusoids))
        candidate_a = np.fromiter(map(lambda x: x.w,candidate_s), dtype=float)

        f_0_estimation = histogram_fundemental(candidate_a)

        if f_0_estimation:
            f_0_sinusoid = next(filter(lambda x: np.isclose(x.w, f_0_estimation), candidate_s))
        else:
            f_0_sinusoid = None

        sinusoid_frames.append(current_sinusoids)
        fundemental_sinusoids.append(f_0_sinusoid)

    # print(f'sinusoid frames size {len(sinusoid_frames)}')
    # print(f'f_0 sinusoid size {len(fundemental_sinusoids)}')
    #
    # print(sinusoid_frames)
    # print(fundemental_sinusoids)

    ts = np.arange(len(fundemental_sinusoids))
    ws = np.array([s.w if s else 0 for s in fundemental_sinusoids])
    amps = np.array([s.a if s else 0 for s in fundemental_sinusoids])

    ws = ws / np.amax(ws)
    amps = amps / np.amax(amps)

    plt.plot(ts, ws)
    plt.plot(ts, amps)
    plt.show()
    plt.clf()

    # n = 70
    # sinus = sinusoid_frames[n]
    # s_w = list(map(lambda x: x.w, sinus))
    # s_a = list(map(lambda x: x.a, sinus))
    # plt.vlines(fundemental_sinusoids[n].w,0,np.amax(s_a))
    # plt.plot(s_w,s_a)
    # plt.show()
    # plt.clf()
    # n = 80
    # sinus = sinusoid_frames[n]
    # s_w = list(map(lambda x: x.w, sinus))
    # s_a = list(map(lambda x: x.a, sinus))
    # plt.vlines(fundemental_sinusoids[n].w,0,np.amax(s_a))
    # plt.plot(s_w,s_a)
    # plt.show()
    # plt.clf()
    # n = 90
    # sinus = sinusoid_frames[n]
    # s_w = list(map(lambda x: x.w, sinus))
    # s_a = list(map(lambda x: x.a, sinus))
    # plt.vlines(fundemental_sinusoids[n].w,0,np.amax(s_a))
    # plt.plot(s_w,s_a)
    # plt.show()
    # plt.clf()
    # n = 100
    # sinus = sinusoid_frames[n]
    # s_w = list(map(lambda x: x.w, sinus))
    # s_a = list(map(lambda x: x.a, sinus))
    # plt.vlines(fundemental_sinusoids[n].w,0,np.amax(s_a))
    # plt.plot(s_w,s_a)
    # plt.show()
    # plt.clf()
    # n = 110
    # sinus = sinusoid_frames[n]
    # s_w = list(map(lambda x: x.w, sinus))
    # s_a = list(map(lambda x: x.a, sinus))
    # plt.vlines(fundemental_sinusoids[n].w,0,np.amax(s_a))
    # plt.plot(s_w,s_a)
    # plt.show()
    # plt.clf()
    # n = 120
    # sinus = sinusoid_frames[n]
    # s_w = list(map(lambda x: x.w, sinus))
    # s_a = list(map(lambda x: x.a, sinus))
    # plt.vlines(fundemental_sinusoids[n].w,0,np.amax(s_a))
    # plt.plot(s_w,s_a)
    # plt.show()
    # plt.clf()


    # for s in range(len(fundemental_sinusoids)):
    #     w =



        # TODO: consider comb fundemental f_0 analysis with entire frame
        #





    # no fundementals above 5000 hz for sure!
    # no fundmentals below 30
    #
    #
    # upper_w = 5000 / 44100 * 2 * np.pi
    # lower_w = 30 / 44100 * 2 * np.pi
    #
    # track_sinusoids = np.array([s for track in tracks for s in track.sinusoids])
    # print(f'{track_sinusoids.size} vs. {len(tracks)}')
    #
    # fundemental_w_t = [(histogram_fundemental(
    #     np.fromiter(
    #         map(lambda x: x.get_nearest_sinusoid(t).w,
    #             filter(lambda x: x is not None and
    #                              x.is_defined_at_time(t) and
    #                              upper_w > x.get_nearest_sinusoid(t).w > lower_w,
    #                    tracks)
    #             ),
    #         dtype=float)), t)
    #     for t in np.arange(signal.size, step=analysis_step)]
    #
    # fundemental_w_t = list(filter(lambda x: x[0] is not None, fundemental_w_t))
    #
    # fundemental_w_a = [(w, (next(filter(lambda s: s.t == t and np.isclose(s.w, w).all(), track_sinusoids), None)).a) for
    #                    (w, t) in fundemental_w_t]
    #
    # print(fundemental_w_a)
    # fundemental_w = [x[0] * rate / (2 * np.pi) if x[0] is not None else None for x in fundemental_w_a]
    # fundemental_a = [x[1] for x in fundemental_w_a]
    #
    # print(f'None in f_w? {np.where(np.array(fundemental_w)is None)}')
    # print(f'None in f_a? {np.where(np.array(fundemental_a)is None)}')
    #
    # plt.plot(np.arange(len(fundemental_w)), fundemental_w)
    # plt.show()
    # plt.clf()
    # plt.plot(np.arange(len(fundemental_a)), fundemental_a)
    # plt.show()
    #
    # plt.clf()

    ##
    ## SYNTHESIS SECTION
    ##

    synthesis_signal = np.zeros(signal.size, dtype=np.float32)
    print(f'out size {synthesis_signal.size}')

    for track in tracks:
        start_i = track.start_frame()
        series = track.to_time_series()
        synthesis_signal[start_i:start_i + series.size] = synthesis_signal[start_i:start_i + series.size] + series

    print(synthesis_signal.dtype)

    synthesis_envelope = get_time_series_envelope(synthesis_signal, rate)

    plt.clf()
    plt.plot(np.arange(synthesis_signal.size), synthesis_signal, label='synthesis')
    plt.plot(np.arange(envelope.size), envelope, label='orginal envelope')
    plt.plot(np.arange(synthesis_envelope.size), synthesis_envelope, label='synthesis_envelope')
    plt.plot(np.arange(signal.size), signal, label='original')
    plt.legend()
    plt.show()

    print(f'signal length {signal.size}')
    print(f'synthesis signal length {synthesis_signal.size}')

    residual = synthesis_signal - signal
    plt.plot(np.arange(residual.size), residual, label='residual')
    plt.title('residual')
    plt.show()

    wavfile.write(r'C:\Users\elber\Documents\AudioRecordings\sin_rep_output_with_mod.wav', rate, synthesis_signal)
    wavfile.write(r'C:\Users\elber\Documents\AudioRecordings\sin_rep_residual.wav', rate, residual)
