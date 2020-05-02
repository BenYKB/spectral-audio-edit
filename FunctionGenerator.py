import numpy as np

class FunctionGenerator:
    def __init__(self, sample_rate):
        self.SAMPLE_RATE = sample_rate

    def get_time(self, length):
        return np.arange(0, length, 1.0 / self.SAMPLE_RATE)

    def cosine_time_series(self, length, frequency, amplitude=1, phase_angle=0):
        time_values = self.get_time(length) * 2 * np.pi * frequency + phase_angle
        return np.cos(time_values) * amplitude

    def gaussian_time_series(self,length, amplitude, time_constant, offset):
        t_array = self.get_time(length)

        def argument(t):
            -1 * amplitude * ((t - offset) / time_constant) ** 2

        apply_argument = np.vectorize(argument)

        return np.exp(apply_argument(t_array))

    def square_time_series(self, length, amplitude, on_width, on_start):
        time_array = self.get_time(length)

        def square_wave(t):
            return amplitude if on_start < t < on_start + on_width else 0

        func = np.vectorize(square_wave)

        return func(time_array)



    def morlet_complex_time_series(self, length, frequency, time_constant, offset):
        time_array = self.get_time(length)

        def morlet_arg(t):
            return 1j * 2 * np.pi * frequency * (t - offset) - .5 * ((t - offset) / time_constant) ** 2

        argument = np.vectorize(morlet_arg)

        return np.exp(argument(time_array))
