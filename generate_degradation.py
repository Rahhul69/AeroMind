import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration: 30 days, 1000 samples per day (1000 Hz)
days = 30
samples_per_day = 1000
fs = 1000 
t = np.linspace(0, 1, samples_per_day, endpoint=False)

# Frequencies
operating_freq = 10.0 # Normal turbine rotation
fault_freq = 50.0 # The Blade Pitch Error signature

data = []

print("System: Initializing 30-day mechanical degradation simulation...")

for day in range(1, days + 1):
    # 1. Healthy turbine noise & base rotation
    noise = np.random.normal(0, 0.5, samples_per_day)
    base_vibration = 2.0 * np.sin(2 * np.pi * operating_freq * t)
    
    # 2. The Fault - Starts tiny, grows exponentially over 30 days
    fault_amplitude = 0.05 * np.exp(0.15 * day) 
    fault_vibration = fault_amplitude * np.sin(2 * np.pi * fault_freq * t)
    
    # 3. Combine to get the day's total telemetry
    daily_telemetry = base_vibration + fault_vibration + noise
    data.extend(daily_telemetry)

# Save the timeline to CSV for the Deep Learning model
df = pd.DataFrame({'Vibration': data})
df.to_csv('degradation_timeline.csv', index=False)
print("Success: Generated 'degradation_timeline.csv' (30,000 rows).")

# Generate a visual proof for you to inspect
plt.figure(figsize=(10, 4), facecolor='#161b22')
ax = plt.axes()
ax.set_facecolor('#0d1117')
plt.plot(data[:1000], label='Day 1 (Healthy)', color='#2ea043', alpha=0.9)
plt.plot(data[-1000:], label='Day 30 (Critical Fault)', color='#f85149', alpha=0.7)
plt.title('Simulated Mechanical Degradation: Day 1 vs Day 30', color='#c9d1d9')
plt.legend()
plt.savefig('degradation_visual.png', facecolor='#161b22')
print("Success: Saved 'degradation_visual.png'.")