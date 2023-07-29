import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp tin CSV
data = pd.read_csv('T0_ID000104_Walk1.csv', header=None)

# Chuyển dữ liệu thành mảng numpy
samples = data.values.astype(float)

# Biến đổi Fourier
fft_result = np.fft.fft(samples, axis=0)
freq = np.fft.fftfreq(samples.shape[0])

fft_magnitude = np.abs(fft_result)
freq_positive = freq[np.where(freq >= 0)]

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs = axs.flatten()

for i in range(samples.shape[1]):
    axs[i].plot(freq_positive, fft_magnitude[:len(freq_positive), i])
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Magnitude')
    axs[i].set_title(f'Fourier Transform of Column {i+1}')
    axs[i].grid(True)

plt.tight_layout()
plt.show()