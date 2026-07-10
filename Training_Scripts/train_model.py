import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

## 1. DATA LOADING & PREPARATION
## Loads the compiled CSV, separates the math features (X) from the labels (y), and splits 80% for training / 20% for testing.
print("Loading the 65,000 row dataset...")
df = pd.read_csv('../final_ml_dataset.csv') # Adjust path if necessary depending on where you run this

# Separate the mathematical features (X) from the answers (y)
X = df[['Peak_Amplitude', 'Mean_Noise', 'RMS_Energy']]
y = df['Fault_Label']

# Split the data: 80% for studying, 20% for the final exam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 2. MODEL TRAINING
## Initializes the Random Forest engine. Capping max_depth at 10 prevents the AI from memorizing random background noise.
print("Training the Random Forest AI...")
# max_depth=10 stops the AI from over-memorizing the background wind noise
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

## 3. EVALUATION & EXPORT
## Tests the model against the unseen 20% dataset, prints the classification report, and packages the model for the web server.
print("\n--- AI PERFORMANCE REPORT ---")
# Test the model on the 20% of data it has never seen before
y_pred = rf_model.predict(X_test)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', '5-Degree Fault (1)']))

# Serialize and save the trained brain 
joblib.dump(rf_model, '../turbine_classifier.pkl')
print("\nModel saved successfully as 'turbine_classifier.pkl'. Ready for API integration.")