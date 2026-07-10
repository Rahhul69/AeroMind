## 1. SETUP AND CONFIGURATION
## Initializes the Flask application, loads the machine learning model, and defines global constants.
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import scipy.io
import joblib
import os
import tempfile

app = Flask(__name__)

classifier = joblib.load('turbine_classifier.pkl')

CHUNK_SIZE = 10240 
SAMPLING_RATE = 10000 

## 2. WEB INTERFACE ROUTE
## Serves the main web interface for the root URL of the application.
@app.route('/')
def home():
    return render_template('index.html')

## 3. API ENDPOINT AND VALIDATION
## Validates the incoming POST request to ensure a valid file was submitted before processing.
@app.route('/predict', methods=['POST'])
def predict():
    if 'file_live' not in request.files:
        return jsonify({"error": "No file uploaded."})
        
    file = request.files['file_live']
    if file.filename == '':
        return jsonify({"error": "Empty file submitted."})

## 4. DATA INGESTION
## Saves the uploaded file temporarily and extracts vibration data from supported .mat or .csv formats.
    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)

        # DATA INGESTION: Handle .mat and .csv
        if filename.endswith('.mat'):
            mat_data = scipy.io.loadmat(temp_path)
            if 'brng_f_y' not in mat_data:
                return jsonify({"error": "Sensor 'brng_f_y' not found in MAT file."})
            vibration_data = mat_data['brng_f_y'].flatten()
        elif filename.endswith('.csv'):
            df = pd.read_csv(temp_path)
            vibration_data = df.iloc[:, 0].values
        else:
            return jsonify({"error": "Unsupported format. Please upload .mat or .csv."})

## 5. ETL AND FEATURE EXTRACTION
## Isolates the final chunk of data and applies a Fast Fourier Transform to extract statistical features.
        raw_row_count = len(vibration_data)
        if raw_row_count < CHUNK_SIZE:
            return jsonify({"error": f"Dataset too small. Need at least {CHUNK_SIZE} rows."})
            
        compressed_row_count = raw_row_count // CHUNK_SIZE
            
        # Get the final snapshot chunk
        chunk = vibration_data[-CHUNK_SIZE:]
        
        # ETL: Fast Fourier Transform
        fft_vals = np.abs(np.fft.fft(chunk))
        fft_half = fft_vals[:CHUNK_SIZE // 2]
        
        peak_amplitude = float(np.max(fft_half))
        mean_noise = float(np.mean(fft_half))
        rms_energy = float(np.sqrt(np.mean(fft_half**2)))
        
## 6. AI INFERENCE
## Feeds the extracted features into the pre-trained classifier to diagnose potential blade pitch errors.
        # INFERENCE: AI Diagnosis
        features = pd.DataFrame(
            [[peak_amplitude, mean_noise, rms_energy]], 
            columns=['Peak_Amplitude', 'Mean_Noise', 'RMS_Energy']
        )
        prediction = classifier.predict(features)
        is_faulty = bool(prediction[0] == 1)
        
        if is_faulty:
            diagnosis = "CRITICAL FAULT: 5-Degree Blade Pitch Error Detected"
            days_remaining = "0 Days (IMMEDIATE MAINTENANCE REQUIRED)"
        else:
            diagnosis = "HEALTHY: Mechanical Parameters Nominal"
            days_remaining = "30+ Days Estimated"
            
## 7. RESPONSE GENERATION AND CLEANUP
## Calculates graph coordinates, returns all diagnostic results as JSON, and safely removes the temporary file.
        # UI GRAPH COORDINATES
        xf_live = np.linspace(0.0, SAMPLING_RATE/2.0, CHUNK_SIZE//2)
        y_live = (2.0 / CHUNK_SIZE) * fft_half
        
        x_time = np.linspace(0.0, CHUNK_SIZE / SAMPLING_RATE, CHUNK_SIZE)
        y_raw = chunk
            
        return jsonify({
            "raw_rows": f"{raw_row_count:,}",
            "compressed_rows": f"{compressed_row_count:,}",
            "diagnosis": diagnosis,
            "is_faulty": is_faulty,
            "days_remaining": days_remaining,
            "x_axis": xf_live.tolist(),
            "y_live": y_live.tolist(),
            "x_time": x_time.tolist(),
            "y_raw": y_raw.tolist()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server failed to process file: {str(e)}"})
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

## 8. EXECUTION
## Starts the Flask development server on port 5000.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)