

from scipy.fftpack import fft, ifft
from scipy.io import wavfile

n = 0
# importing the sound files
fs, IR1 = wavfile.read('C:\\Users\\Benjamin\\Desktop\\MtRoyalStairs.wav')
fs1, S1 = wavfile.read('C:\\Users\\Benjamin\\Desktop\\SSexcerpt.wav')

h1 = IR1.T
x1 = S1.T[0]
m = len(x1) - len(h1)

samples = len(h1)
if len(x1) < samples:
    samples = len(x1)
# taking FFT
Fh1 = fft(h1[:samples])
Fx1 = fft(x1[:samples])

Fy1 = []
# multiplication in frequency domain
while n < samples:
    Fy1.append(Fh1[n]*Fx1[n])
    n += 1
# taking inverse FFT
y1 = ifft(Fy1).real
test = abs(ifft(Fx1))
# exporting file with convolution reverb
wavfile.write('C:\\Users\\Benjamin\\Desktop\\outputoboe.wav', 48000, y1)
