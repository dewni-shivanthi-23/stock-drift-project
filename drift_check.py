import yfinance as yf
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")

# New data
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Return'] > 0).astype(int)

data['MA5'] = data['Close'].rolling(5).mean()
data['MA10'] = data['Close'].rolling(10).mean()

data = data.dropna()

X = data[['MA5', 'MA10']]
y = data['Target']

preds = model.predict(X)
acc = accuracy_score(y, preds)

print("Accuracy:", acc)

if acc < 0.55:
    print("⚠️ Drift detected!")
else:
    print("✅ Model stable")