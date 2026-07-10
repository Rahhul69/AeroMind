import os
import scipy.io
import numpy as np
import pandas as pd

# --- PIPELINE CONFIGURATION ---
DATA_DIR = 'data'
TARGET_SENSOR = 'brng_f_y'  # Front bearing lateral vibration
CHUNK_SIZE = 10240          # Standard power of 2 for optimal FFT math

def process_folder(folder_name, label_value):
    folder_path = os.path.join(DATA_DIR, folder_name)
    extracted_features = []
    
    print(f"Processing folder: {folder_name} | Assigning ML Label: {label_value}")
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"  -> Warning: Folder '{folder_path}' not found. Skipping.")
        return extracted_features

    # Loop through every .mat file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                mat_data = scipy.io.loadmat(file_path)
                
                # Verify our target sensor is actually in this file
                if TARGET_SENSOR in mat_data:
                    # Flatten the array from 2D to 1D
                    vibration_data = mat_data[TARGET_SENSOR].flatten()
                    
                    # Slice the massive array into small 10,240-row chunks
                    for i in range(0, len(vibration_data) - CHUNK_SIZE, CHUNK_SIZE):
                        chunk = vibration_data[i : i + CHUNK_SIZE]
                        
                        # Execute Fast Fourier Transform (FFT)
                        fft_vals = np.abs(np.fft.fft(chunk))
                        fft_half = fft_vals[:CHUNK_SIZE // 2] # Keep only positive frequencies
                        
                        # Extract the mathematical features for the AI
                        peak_amplitude = np.max(fft_half)
                        mean_noise = np.mean(fft_half)
                        rms_energy = np.sqrt(np.mean(fft_half**2))
                        
                        # Store the row
                        extracted_features.append({
                            'Peak_Amplitude': peak_amplitude,
                            'Mean_Noise': mean_noise,
                            'RMS_Energy': rms_energy,
                            'Fault_Label': label_value
                        })
            except Exception as e:
                print(f"  -> Error reading {filename}: {e}")
                
    return extracted_features

# --- EXECUTE THE ETL PROCESS ---
print("Starting ETL Pipeline...")

# Label 0 for baseline mechanical slop, Label 1 for active blade pitch error
healthy_data = process_folder('Healthy', 0)
faulty_data = process_folder('5_Degrees', 1)

# Combine the two datasets
all_data = healthy_data + faulty_data

if len(all_data) > 0:
    # Convert to a Pandas DataFrame and save to a clean CSV
    df = pd.DataFrame(all_data)
    df.to_csv('final_ml_dataset.csv', index=False)
    
    print("\n--- ETL COMPLETE ---")
    print(f"Total feature rows generated: {len(df)}")
    print("Exported successfully to: final_ml_dataset.csv")
else:
    print("\nETL FAILED: No data was processed. Check your folder names and paths.")