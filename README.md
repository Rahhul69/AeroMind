# AeroMind Cloud
**Predictive Edge-to-Cloud Diagnostic Engine**

AeroMind is a machine learning pipeline designed to ingest massive, high-velocity physical telemetry from edge nodes (wind turbines), distill the signal using mathematical transformations, and diagnose critical mechanical faults in real-time.

## ⚙️ The Architecture

This system relies on a two-stage edge-to-cloud pipeline:

1. **Stage 1: ETL Data Compression & Distillation**
   The backend ingestion adapter accepts high-frequency edge-node telemetry (`.MAT` or `.CSV`). It isolates the target vibration arrays and executes a Fast Fourier Transform (FFT). This process aggressively compresses the raw time-series data and extracts three pure mathematical features: Peak Amplitude, RMS Energy, and Mean Noise.
   
2. **Stage 2: AI Diagnostic Inference**
   A highly optimized Scikit-Learn Random Forest Classifier (`turbine_classifier.pkl`) analyzes the distilled FFT features. It isolates the specific mechanical harmonic signatures (e.g., separating normal background wind slop from a critical 5-Degree Blade Pitch Error) and outputs an immediate binary classification.

## 🛠️ Tech Stack
* **Backend:** Python, Flask, Pandas, NumPy, SciPy (Data Ingestion)
* **Machine Learning:** Scikit-Learn, Joblib
* **Frontend:** HTML5, CSS3, JavaScript (Single Page Application)
* **Visualization:** Chart.js (Real-time Frequency Domain rendering)

## 🚀 Installation & Boot Sequence

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Rahhul69/AeroMind.git](https://github.com/Rahhul69/AeroMind.git)
   cd AeroMind

2. Activate the Virtual Environment (Windows PowerShell):
   ```PowerShell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1

3. Install Core Dependencies:
   ```Bash
   pip install -r requirements.txt

4. Launch the Cloud Server:
   ```Bash
   python app.py

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to access the AeroMind dashboard.