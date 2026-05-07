from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.fft import fft
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hides TensorFlow warnings on the server
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Both AI Brains
print("System: Booting Diagnostic Classifier...")
classifier = joblib.load('turbine_classifier.pkl')

print("System: Booting Predictive LSTM Forecaster...")
forecaster = load_model('predictive_forecaster.keras')

chunk_size = 1000
fs = 1000 
CRITICAL_VARIANCE_THRESHOLD = 0.8 # The mathematical breaking point

def process_and_predict(data_array):
    features = []
    variances = []
    
    # 1. Extract Features for the Classifier & Forecaster
    for i in range(0, len(data_array) - chunk_size, chunk_size):
        chunk = data_array[i:i + chunk_size]
        yf = np.abs(fft(chunk)[0:chunk_size//2])
        var_val = np.var(yf)
        
        features.append([np.max(yf), np.mean(yf), var_val])
        variances.append(var_val)
    
    if not features:
        return "ERROR", False, "N/A"

    # 2. Ask the Random Forest if it is broken RIGHT NOW
    predictions = classifier.predict(features)
    fault_ratio = np.mean(predictions)
    is_faulty = bool(fault_ratio > 0.10)
    
    diagnosis = "CRITICAL FAULT DETECTED: Blade Pitch Error" if is_faulty else "HEALTHY: Turbine Operating Normally"

    # 3. Ask the LSTM to predict the future wear-and-tear
    # We take the recent variance trend, scale it roughly, and ask the LSTM
    recent_trend = np.array(variances[-50:]).reshape(1, -1, 1) 
    
    # Pad with zeros if the file is too short to have 50 chunks
    if recent_trend.shape[1] < 50:
        pad_width = 50 - recent_trend.shape[1]
        recent_trend = np.pad(recent_trend, ((0,0), (pad_width, 0), (0,0)), mode='constant')

    # Normalize roughly to match training scale
    recent_trend = recent_trend / np.max(recent_trend) if np.max(recent_trend) > 0 else recent_trend
    
    next_step_degradation = forecaster.predict(recent_trend, verbose=0)[0][0]
    
    # 4. Calculate estimated days remaining based on the LSTM's trajectory
    if is_faulty:
        days_remaining = "0 Days (IMMEDIATE MAINTENANCE REQUIRED)"
    else:
        # Inverse calculate remaining lifespan based on the degradation curve
        health_score = 1.0 - min(next_step_degradation, 1.0)
        estimated_days = int(health_score * 30) # 30 day max lifespan
        days_remaining = f"{max(1, estimated_days)} Days Estimated"

    return diagnosis, is_faulty, days_remaining

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    baseline_file = request.files.get('file_baseline')
    live_file = request.files.get('file_live')

    if not baseline_file or not live_file or baseline_file.filename == '' or live_file.filename == '':
        return jsonify({"error": "Please upload both Baseline and Live files."})

    try:
        # Process Baseline
        df_base = pd.read_csv(baseline_file)
        base_chunk = df_base.iloc[:1000, 0].dropna().values
        yf_base = np.abs(fft(base_chunk)[0:500])
        xf_base = np.linspace(0.0, fs/2.0, 500)

        # Process Live Data
        df_live = pd.read_csv(live_file)
        live_array = df_live.iloc[:, 0].dropna().values
        live_chunk = live_array[:1000]
        yf_live = np.abs(fft(live_chunk)[0:500])
        xf_live = np.linspace(0.0, fs/2.0, 500)

        # Run Dual-Engine AI Diagnosis
        diagnosis, is_faulty, days_remaining = process_and_predict(live_array)

        return jsonify({
            "diagnosis": diagnosis,
            "is_faulty": is_faulty,
            "days_remaining": days_remaining,
            "x_axis": xf_live.tolist(),
            "y_base": (2.0/1000 * yf_base).tolist(),
            "y_live": (2.0/1000 * yf_live).tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)