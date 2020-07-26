import numpy as np
from scipy.io import wavfile

class PhaseVocoder:
    def __init__(self):
        pass




if __name__ == "__main__":
    N = 2048
    sr, s = wavfile.read(r'C:\Users\elber\Documents\AudioRecordings\input_signal.wav')
    signal_1 = np.array(s, dtype=np.float32)




