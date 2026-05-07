# AeroMind: Predictive AI Engine
**Advanced Time-Series Analytics for Wind Energy Systems**

AeroMind is a dual-engine machine learning pipeline designed to ingest live, high-velocity physical telemetry from wind turbines, diagnose current mechanical states, and predict future catastrophic failures using frequency domain modeling.

## 🧠 The Dual-Engine Architecture

This system relies on a two-stage artificial intelligence pipeline:

1. **The Fast Fourier Transform (FFT) Pipeline**
   Raw time-series vibration data is ingested and converted into the frequency domain. This isolates the specific mechanical harmonic signatures of a turbine (e.g., separating normal wind noise from a 50Hz Blade Pitch Error).
   
2. **The Diagnostician (Random Forest Classifier)**
   A lightweight, highly optimized Scikit-Learn model (`turbine_classifier.pkl`). It analyzes the peak frequency amplitudes and rolling variance of the FFT output to provide an instantaneous binary classification: **Healthy** or **Critical Fault**.

3. **The Forecaster (LSTM Neural Network)**
   A Deep Learning sequence model built in TensorFlow (`predictive_forecaster.keras`). Instead of looking at current state, it analyzes the micro-degradation trend (rolling variance) over a simulated 30-day timeline to predict the exact remaining operational lifespan of the turbine before complete mechanical failure.

## ⚙️ Tech Stack
* **Backend:** Python, Flask, NumPy, SciPy, Pandas
* **Machine Learning:** TensorFlow/Keras, Scikit-Learn, Joblib
* **Frontend:** HTML5, CSS3, JavaScript, Chart.js (for real-time X-Ray visualization)

## 🚀 Installation & Boot Sequence

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rahhul69/AeroMind.git
   cd AeroMind