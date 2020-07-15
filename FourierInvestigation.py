import numpy as np
import matplotlib.pyplot as plt

N = 20
times = np.arange(N)
sine_1 = np.sin(times*np.pi/N)
sine_5 = np.sin(times*2*np.pi/N)


plt.plot(times, sine_1)
plt.plot(times, sine_5)

plt.show()