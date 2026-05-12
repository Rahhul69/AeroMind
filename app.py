from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import joblib
import os
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load the trained brains into memory on boot
classifier = joblib.load('turbine_classifier.pkl')
forecaster = load_model('predictive_forecaster.keras')

CHUNK_SIZE = 1000 
SAMPLING_RATE = 1000 

def process_and_predict(vibration_data):
    x_features, lstm_sequence = [], []
    
    # Process the incoming live stream in 1-second chunks
    for i in range(0, len(vibration_data), CHUNK_SIZE):
        chunk = vibration_data[i:i+CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE: break
            
        # Re-apply the math used during training
        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk)))
        
        N = len(chunk)
        yf = fft(chunk)
        xf = fftfreq(N, 1/SAMPLING_RATE)
        dominant_freq = float(xf[np.argmax(np.abs(yf[0:N//2]))])
        
        x_features.append([rms, peak, dominant_freq])
        lstm_sequence.append([rms, peak])
        
    if not x_features: return "ERROR", False, "N/A"

    # Step 1: Random Forest Diagnosis
    # Grab the very last second of data and flatten it for the AI
    latest_features = np.array(x_features[-1]).reshape(1, 3)
    rf_prediction = classifier.predict(latest_features)
    is_faulty = bool(np.ravel(rf_prediction)[0] == 1)
    
    diagnosis = "CRITICAL FAULT: 50Hz Harmonic Spike" if is_faulty else "HEALTHY: Normal Parameters"

    # Step 2: LSTM Lifespan Prediction
    # Feed the entire sequence to the LSTM to analyze the degradation trend over time
    sequence_array = np.array(lstm_sequence).reshape(1, len(lstm_sequence), 2)
    raw_lstm_output = forecaster.predict(sequence_array, verbose=0)
    
    # Steamroller: Flatten the timeline and grab the final health score
    next_step = float(np.ravel(raw_lstm_output)[-1])
    
    if is_faulty:
        days_remaining = "0 Days (IMMEDIATE MAINTENANCE)"
    else:
        # Scale the 0.0 - 1.0 score against a hypothetical 30-day lifespan
        health_score = max(0.0, min(next_step, 1.0))
        days_remaining = f"{max(1, int(health_score * 30))} Days Estimated"

    return diagnosis, is_faulty, days_remaining

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive the CSV payload from the Edge Node
    live_file = request.files.get('file_live')
    if not live_file: return jsonify({"error": "Upload telemetry CSV."})

    try:
        vibration_array = pd.read_csv(live_file)['vibration_g'].values
        diagnosis, is_faulty, days_remaining = process_and_predict(vibration_array)

        # Generate the FFT graph data for the frontend UI
        last_chunk = vibration_array[-CHUNK_SIZE:]
        yf_live = np.abs(fft(last_chunk)[0:CHUNK_SIZE//2])
        xf_live = np.linspace(0.0, SAMPLING_RATE/2.0, CHUNK_SIZE//2)

        return jsonify({
            "diagnosis": diagnosis, "is_faulty": is_faulty, "days_remaining": days_remaining,
            "x_axis": xf_live.tolist(), "y_live": (2.0/CHUNK_SIZE * yf_live).tolist()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)