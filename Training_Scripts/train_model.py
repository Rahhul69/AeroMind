import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

fs = 1000  # Sampling frequency

# 1. Load the NEW Augmented Data
path_healthy = r"D:\Rahhhul\PBL\Dataset\healthy_augmented.csv"
path_faulty = r"D:\Rahhhul\PBL\Dataset\faulty_augmented.csv"

print("Loading massive datasets...")
y_healthy = pd.read_csv(path_healthy).iloc[:, 0].dropna().values
y_faulty = pd.read_csv(path_faulty).iloc[:, 0].dropna().values

# 2. Feature Extraction (The "Chunking" Process)
def extract_features(data_array, label, chunk_size=1000):
    features = []
    # Break the massive array into 1-second chunks (1000 readings)
    for i in range(0, len(data_array) - chunk_size, chunk_size):
        chunk = data_array[i:i + chunk_size]
        
        # Apply FFT to the chunk
        yf = np.abs(fft(chunk)[0:chunk_size//2])
        
        # Extract key mathematical features from the FFT
        max_amplitude = np.max(yf)
        mean_amplitude = np.mean(yf)
        variance = np.var(yf)
        
        # Store the features and the label (0 = Healthy, 1 = Faulty)
        features.append([max_amplitude, mean_amplitude, variance, label])
        
    return features

# Create our training dataset
print("Extracting features from 100,000 rows of data...")
healthy_data = extract_features(y_healthy, label=0)
faulty_data = extract_features(y_faulty, label=1)

# Combine and convert to a Pandas DataFrame
dataset = pd.DataFrame(healthy_data + faulty_data, columns=['Max_Amp', 'Mean_Amp', 'Variance', 'Status'])

X = dataset[['Max_Amp', 'Mean_Amp', 'Variance']] # The Inputs
y = dataset['Status']                            # The Output (0 or 1)

# 3. Train the Machine Learning Model
print("Training the Random Forest Classifier...")
# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the AI
model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
model.fit(X_train, y_train)

# 4. Test the Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n--- Model Training Complete ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 5. Export the Brain
joblib.dump(model, 'turbine_classifier.pkl')
print("\nModel successfully saved as 'turbine_classifier.pkl'")