import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Data Load 
df = pd.read_csv('AAPL_data.csv', index_col=0)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()
data = df.filter(['Close']).values

# 2. Data Scaling 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

print(f"Data scaled successfully. Shape: {scaled_data.shape}")

# 3. Training Data 
training_data_len = int(np.ceil(len(data) * .8))
train_data = scaled_data[0:int(training_data_len), :]

x_train, y_train = [], []

# Make 60 days window
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. Create LSTM Model 
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# 5. Model Compiling
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Training 
print("Training started...")
model.fit(x_train, y_train, batch_size=32, epochs=10)
print("Training completed!")

# Model saving 
model.save('models/stock_model.h5')
print("Model saved in models folder.")

# 7. Testing 
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = data[training_data_len:, :] 

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 8. Get predictions 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) 

# 9. Draw Graph 
train = df[:training_data_len]
valid = df[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Stock Price Predictor - Apple Inc.')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual Value', 'Predictions'], loc='lower right')
plt.show()