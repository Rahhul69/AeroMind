import os
import scipy.io
import numpy as np
import pandas as pd

## 1. PIPELINE CONFIGURATION
## Sets global variables, target sensor (front bearing lateral vibration), and locks chunk size for optimal FFT math.
DATA_DIR = 'data'
TARGET_SENSOR = 'brng_f_y'  
CHUNK_SIZE = 10240          

## 2. DATA INGESTION & PIPELINE LOGIC
## Scans folders for .mat files, flattens the 2D arrays, and slices them into 10,240-row chunks.
def process_folder(folder_name, label_value):
    folder_path = os.path.join(DATA_DIR, folder_name)
    extracted_features = []
    
    print(f"Processing folder: {folder_name} | Assigning ML Label: {label_value}")
    
    if not os.path.exists(folder_path):
        print(f"  -> Warning: Folder '{folder_path}' not found. Skipping.")
        return extracted_features

    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                mat_data = scipy.io.loadmat(file_path)
                
                if TARGET_SENSOR in mat_data:
                    vibration_data = mat_data[TARGET_SENSOR].flatten()
                    
                    for i in range(0, len(vibration_data) - CHUNK_SIZE, CHUNK_SIZE):
                        chunk = vibration_data[i : i + CHUNK_SIZE]
                        
                        ## 2a. FAST FOURIER TRANSFORM (FFT) MATH
                        ## Converts time-domain wave into frequency domain and extracts Peak, Mean, and RMS features.
                        fft_vals = np.abs(np.fft.fft(chunk))
                        fft_half = fft_vals[:CHUNK_SIZE // 2] 
                        
                        peak_amplitude = np.max(fft_half)
                        mean_noise = np.mean(fft_half)
                        rms_energy = np.sqrt(np.mean(fft_half**2))
                        
                        extracted_features.append({
                            'Peak_Amplitude': peak_amplitude,
                            'Mean_Noise': mean_noise,
                            'RMS_Energy': rms_energy,
                            'Fault_Label': label_value
                        })
            except Exception as e:
                print(f"  -> Error reading {filename}: {e}")
                
    return extracted_features

## 3. EXECUTE THE ETL PROCESS
## Runs the pipeline on both datasets, assigns 0 (Healthy) or 1 (Faulty), and exports to a final CSV.
print("Starting ETL Pipeline...")

healthy_data = process_folder('Healthy', 0)
faulty_data = process_folder('5_Degrees', 1)

all_data = healthy_data + faulty_data

if len(all_data) > 0:
    df = pd.DataFrame(all_data)
    df.to_csv('final_ml_dataset.csv', index=False)
    
    print("\n--- ETL COMPLETE ---")
    print(f"Total feature rows generated: {len(df)}")
    print("Exported successfully to: final_ml_dataset.csv")
else:
    print("\nETL FAILED: No data was processed. Check your folder names and paths.")