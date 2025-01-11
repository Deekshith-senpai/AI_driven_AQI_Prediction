import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('station_hour.csv')
data.fillna(method='ffill', inplace=True)

# Add lag and rolling features
def add_features(df, target_column, lags, rolling_window=3):
    for lag in lags:
        df[f'{target_column}_lag{lag}'] = df[target_column].shift(lag)
    df[f'{target_column}_rolling_mean'] = df[target_column].rolling(window=rolling_window).mean()
    df[f'{target_column}_rolling_std'] = df[target_column].rolling(window=rolling_window).std()
    df.dropna(inplace=True)
    return df

lags = [1, 2, 3, 6, 12, 24]
data = add_features(data, target_column='AQI', lags=lags)

# Split features and target
X = data.drop(columns=['AQI'])
y = data['AQI']  # Assuming AQI is the target column

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a more complex LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Reshape for LSTM
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create and train the LSTM model
lstm_model = create_lstm_model((X_train.shape[1], 1))
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)

# Train individual models (Random Forest, Gradient Boosting)
rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
gb_model.fit(X_train, y_train)

# Get predictions from all models
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# Combine predictions using a more complex meta-model
meta_features = np.column_stack([y_pred_rf, y_pred_gb, y_pred_lstm])
meta_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
meta_model.fit(meta_features, y_test)

# Final predictions
meta_predictions = meta_model.predict(meta_features)

# Evaluate the final ensemble model
r2 = r2_score(y_test, meta_predictions)
rmse = np.sqrt(mean_squared_error(y_test, meta_predictions))
print(f"Ensemble R² Score: {r2:.4f}")
print(f"Ensemble RMSE: {rmse:.4f}")
#gui
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np

# Placeholder for the model (use your trained model here)
class MockModel:
    def predict(self, inputs):
        return [np.random.uniform(50, 200)]  # Simulated prediction

model = MockModel()

# Updated predict function
def predict_aqi():
    try:
        inputs = [float(entry.get()) for entry in entries]
        prediction = model.predict([inputs])[0]
        result_label.config(text=f"Predicted AQI: {prediction:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all inputs.")

# Function to load data
def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        global data
        data = pd.read_csv(file_path)
        messagebox.showinfo("File Loaded", "Dataset successfully loaded.")

# GUI setup
app = tk.Tk()
app.title("AQI Prediction App")

frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Feature columns
input_labels = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
entries = []

for i, label in enumerate(input_labels):
    ttk.Label(frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
    entry = ttk.Entry(frame)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

predict_button = ttk.Button(frame, text="Predict AQI", command=predict_aqi)
predict_button.grid(row=len(input_labels), column=0, columnspan=2, pady=10)

load_button = ttk.Button(frame, text="Load Dataset", command=load_data)
load_button.grid(row=len(input_labels) + 1, column=0, columnspan=2, pady=10)

result_label = ttk.Label(frame, text="", foreground="blue")
result_label.grid(row=len(input_labels) + 2, column=0, columnspan=2, pady=10)

app.mainloop()

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', alpha=0.7)
plt.plot(meta_predictions[:100], label='Ensemble Predicted', alpha=0.7)
plt.title('Actual vs Ensemble Predicted AQI Levels (First 100 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('AQI Level')
plt.legend()
plt.grid(True)
plt.show()