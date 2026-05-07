import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

print("System: Loading 30-day telemetry timeline...")
df = pd.read_csv('degradation_timeline.csv')

# 1. Extract the "Wear-and-Tear" Trend (Rolling Variance)
trend = df['Vibration'].rolling(window=1000).var().dropna().values.reshape(-1, 1)

# 2. Scale data between 0 and 1 (Neural Networks require this)
scaler = MinMaxScaler()
trend_scaled = scaler.fit_transform(trend)

# 3. Build Time-Series Sequences (Look at past 50 points to predict the next 1)
X, y = [], []
for i in range(50, len(trend_scaled)):
    X.append(trend_scaled[i-50:i, 0])
    y.append(trend_scaled[i, 0])
    
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Build the Deep Learning LSTM Brain
print("System: Compiling LSTM Neural Network...")
model = Sequential([
    LSTM(32, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1) # Predicts the single next degradation value
])
model.compile(optimizer='adam', loss='mse')

# 5. Train the Model
print("System: Training Predictive Forecaster (This will take a moment)...")
model.fit(X, y, epochs=5, batch_size=32, verbose=1)

# 6. Save the Brain
model.save('predictive_forecaster.keras')
print("Success: Deep Learning Brain saved as 'predictive_forecaster.keras'.")