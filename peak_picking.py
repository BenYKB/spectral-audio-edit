import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def get_critical_band(b):
    '''
    Critical band centers as suggested in https://www.ee.columbia.edu/~dpwe/papers/Klap06-multif0.pdf
    :param b: band number (typically 1-30)
    :return: band center in hertz
    '''
    return 299 * (10 ** ((b+1) / 21.4) - 1)


def whiten_spectrum(spectrum, rate, is_rfft=True):
    fft_length = spectrum.size * 2 if is_rfft else spectrum.size

    critical_bands = get_critical_band(np.arange(-1,30))
    nyquist = rate // 2
    critical_bands = critical_bands[np.where(critical_bands < nyquist)]

    b_nums = np.arange(1, critical_bands.size - 1)
    bands = np.rint(np.array([[critical_bands[b-1], critical_bands[b], critical_bands[b+1]] for b in b_nums]) * fft_length / rate).astype(int)

    sigmas = []

    for band in bands:
        values = np.array([0,1,0])
        start = band[0].astype(int)
        end = band[2].astype(int)
        band_response = scipy.interpolate.interp1d(band, values, kind='linear')

        sigma = (1/fft_length
                 * np.sum(band_response(np.arange(start, end))
                 * spectrum[start:end] ** 2))**0.5
        sigmas.append(sigma)

    v = 1.0/3.0
    print(f'sigmas {sigmas}')

    whitening_factors = np.array(sigmas) ** (v - 1)
    factor_centers = np.rint(critical_bands[1:critical_bands.size-1] * fft_length / rate).astype(int)

    whitening_function = scipy.interpolate.interp1d(factor_centers, whitening_factors, kind='linear', bounds_error=False, fill_value=1)


    return whitening_function(np.arange(spectrum.size)) * spectrum







def harmonic_comb_fundemental(ft_sig, comb_peaks=7, interpolate=True, lowerbound_index=None, upperbound_index=None, n_iterations=5, rate=44100):
    '''
    Uses Maximal likelihood algorithm as described by https://ccrma.stanford.edu/~jos/sasp/Getting_Closer_Maximum_Likelihood.html

    :param ft_sig:
    :return:
    '''

    ft_sig = ft_sig / np.sum(ft_sig)

    X = whiten_spectrum(np.abs(ft_sig), rate)
    signal_length = X.size
    eps = np.average(X[0:signal_length//3])
    comb = np.arange(1, comb_peaks+1)
    coarse_f_estimates = []

    if not lowerbound_index:
        lowerbound_index = 1
    if not upperbound_index:
        upperbound_index = signal_length//2

    indexed_f = np.arange(lowerbound_index, upperbound_index)

    for f_i in indexed_f:
        product = 1.0
        for h_k in f_i*comb:
            product *= X[h_k] + eps if h_k < signal_length else eps

        coarse_f_estimates.append(product)

    coarse_f_estimates = np.array(coarse_f_estimates)
    likely_indexed_f = indexed_f[np.argmax(coarse_f_estimates)]

    plt.plot(indexed_f, np.log(coarse_f_estimates))

    eps_array = eps* np.ones(X.size)
    plt.plot(np.arange(X.size), np.log(X + eps_array))
    plt.vlines(likely_indexed_f*comb, 5, 10)
    plt.hlines(np.log(eps), 0, 100)
    plt.show()


    if not interpolate:
        n = signal_length // likely_indexed_f
        return likely_indexed_f, np.take(X, likely_indexed_f*np.arange(1, n))

    X_interp = scipy.interpolate.interp1d(np.arange(signal_length), X, kind='cubic', bounds_error=False, fill_value=(0,0))

    eps_a = eps * np.ones(comb.size)

    def likelihood(f_i):
        return np.product(X_interp(f_i*comb)+eps_a)

    f_0 = likely_indexed_f
    sieve_size = 1.0
    divisions = 10

    fs = []
    ls = []

    for n in range(n_iterations):
        fs.append(f_0)
        ls.append(likelihood(f_0))
        f_sieve = np.linspace(f_0-sieve_size, f_0+sieve_size, num=divisions)
        f_0 = f_sieve[np.argmax(np.array([likelihood(f) for f in f_sieve]))]
        sieve_size = 2 * sieve_size / divisions

    # ls = np.array(ls)
    # ns = np.linspace(0,700,num=700*10)
    # xs = np.log(X_interp(ns))
    # plt.plot(ns, xs, xs/np.amax(xs))
    # plt.scatter(fs, np.arange(len(fs))/len(fs))
    # plt.scatter(fs, ls/np.max(ls))
    # plt.vlines(f_0*comb,-2,5)
    # plt.show()

    n = signal_length//likely_indexed_f
    long_comb = f_0 * np.arange(1, n+1)

    harmonics = X_interp(long_comb)
    return f_0, harmonics/np.sum(harmonics)


def histogram_fundemental(peaks, selectivity=2):
    '''
    Estimates fundemental frequency from harmonic peaks
    Histogram Method: https://ccrma.stanford.edu/~jos/pasp/Fundamental_Frequency_Estimation.html
    :param peaks: numpy array
    :param selectivity: greater or equal to 2. How small the histogram bins are
    :return: most likely fundemental of the passed peaks
    '''
    if peaks.size < 2:
        return None

    peaks.sort()

    deltas = peaks[1:] - peaks[:-1]

    histogram = np.zeros(peaks.size)

    for f in deltas:
        if abs(peaks[0] - f) < deltas[0] / selectivity:
            histogram[0] += 1
        elif abs(peaks[-1] - f) < deltas[-1] / selectivity:
            histogram[-1] += 1
        elif f < peaks[0] or f > peaks[-1]:
            pass
        else:
            for i in range(1, len(peaks) - 1):
                if peaks[i] - deltas[i - 1] / selectivity < f < peaks[i] + deltas[i] / selectivity:
                    histogram[i] += 1

    i_max = np.argmax(histogram)

    return peaks[i_max] if np.isscalar(i_max) else peaks[i_max[0]]

def parabolic_peak(p1, p2, p3):
    x = [p1[0], p2[0], p3[0]]
    x_array = np.array([[x_p ** 2, x_p, 1] for x_p in x])

    y_vec = np.array([p1[1], p2[1], p3[1]])

    c, _, _, _ = np.linalg.lstsq(x_array, y_vec,rcond=None)

    x_p = -1 * c[1] / (2 * c[0])
    y_p = c[0] * x_p ** 2 + c[1] * x_p + c[2]


    return x_p, y_p


def find_peaks(frame, radius=2, minimum_height=0.0, maximum_number=None):
    """
    Finds peaks specififying a peak as the largest value in the nearest 5 values
    :param frame: periodigram frame
    :return:
    """
    peak_indicies = []
    n = frame.size

    max_in_group = np.amax(frame[0:radius * 2 + 1])

    for i in range(0, radius + 1):
        if frame[i] == max_in_group and frame[i] > minimum_height:
            peak_indicies.append(i)

    for i in range(radius + 1, n - (radius + 1)):
        if frame[i - (radius + 1)] == max_in_group:
            max_in_group = np.amax(frame[i - radius:i + (radius + 1)])
        elif frame[i + radius] > max_in_group:
            max_in_group = frame[i + radius]

        if frame[i] == max_in_group and frame[i] > minimum_height:
            peak_indicies.append(i)

    max_in_group = np.amax(frame[n - (radius * 2 + 1):n])

    for i in range(n - (radius + 1), n):
        if frame[i] == max_in_group and frame[i] > minimum_height:
            peak_indicies.append(i)

    if maximum_number and len(peak_indicies) > maximum_number:
        i_a = [(i, frame[i]) for i in peak_indicies]
        i_a.sort(key=lambda index_amplitude: index_amplitude[1], reverse=True)
        peak_indicies = [e[0] for e in i_a[0:maximum_number]]
        peak_indicies.sort()

    return np.array(peak_indicies, dtype=np.int32)

    # TODO: implement percentage of max peak selection