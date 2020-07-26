import numpy as np
from scipy.io import wavfile

n = 0
# importing the sound files
fs, IR1 = wavfile.read(r'C:\Users\elber\Documents\AudioRecordings\cyo_ir.wav')
fs1, S1 = wavfile.read(r'C:\Users\elber\Documents\AudioRecordings\input_signal.wav')

assert(fs == fs1)
print(fs)

t_impulse_response = np.array(IR1, dtype=np.float32)
t_signal = np.array(S1, dtype=np.float32)

print(t_signal)
print(np.max(t_signal))

signal_power = np.trapz((t_signal ** 2)) ** 0.5
ir_power = np.trapz((t_impulse_response ** 2)) ** 0.5
print(ir_power)

signal_strength = 50
t_signal = t_signal / signal_power * signal_strength
t_impulse_response = t_impulse_response / ir_power

t_signal = np.pad(t_signal, (0, t_impulse_response.size))
padding = t_signal.size - t_impulse_response.size
t_impulse_response = np.pad(t_impulse_response, (0, padding))

f_impulse_response = np.fft.fft(t_impulse_response)
f_signal = np.fft.fft(t_signal)

f_result = f_impulse_response * f_signal
t_result = np.real(np.fft.ifft(f_result, f_signal.size))

wavfile.write(r'C:\Users\elber\Documents\AudioRecordings\convolutionOutput.wav', fs, t_result)
