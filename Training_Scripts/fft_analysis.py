import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

fs = 1000  

# Pointing to the new massive, clean datasets
path_healthy = r"D:\Rahhhul\PBL\Dataset\healthy_augmented.csv"
path_faulty = r"D:\Rahhhul\PBL\Dataset\faulty_augmented.csv"

# 1. Load Data (Stripped down to exactly one line per file)
y_healthy = pd.read_csv(path_healthy).iloc[:, 0].dropna().values
y_faulty = pd.read_csv(path_faulty).iloc[:, 0].dropna().values

N_h = len(y_healthy)
N_f = len(y_faulty)

print(f"Data Extracted: {N_h} healthy rows | {N_f} faulty rows")

if N_h == 0 or N_f == 0:
    print("\n--- ERROR: Missing data ---")
else:
    # 2. Calculate FFT for Healthy
    xf_h = fftfreq(N_h, 1 / fs)[:N_h//2] 
    yf_h = fft(y_healthy)
    yf_h_plot = 2.0/N_h * np.abs(yf_h[0:N_h//2]) 

    # Calculate FFT for Faulty
    xf_f = fftfreq(N_f, 1 / fs)[:N_f//2] 
    yf_f = fft(y_faulty)
    yf_f_plot = 2.0/N_f * np.abs(yf_f[0:N_f//2])

    # 3. Draw the Charts
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(xf_h, yf_h_plot, color='green')
    plt.title('Healthy Turbine (100k Rows)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(xf_f, yf_f_plot, color='red')
    plt.title('Blade Twist Fault (100k Rows)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()