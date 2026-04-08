import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Download data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Features
data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Return'] > 0).astype(int)

data['MA5'] = data['Close'].rolling(5).mean()
data['MA10'] = data['Close'].rolling(10).mean()

data = data.dropna()

# Train model
X = data[['MA5', 'MA10']]
y = data['Target']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained!")