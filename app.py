# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import joblib

# st.title("📈 Stock Drift Detector")

# ticker = st.text_input("Enter stock (AAPL)", "AAPL")

# data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# data['Return'] = data['Close'].pct_change()
# data['Target'] = (data['Return'] > 0).astype(int)

# data['MA5'] = data['Close'].rolling(5).mean()
# data['MA10'] = data['Close'].rolling(10).mean()

# data = data.dropna()

# model = joblib.load("model.pkl")

# X = data[['MA5', 'MA10']]
# y = data['Target']

# preds = model.predict(X)
# acc = (preds == y).mean()

# st.write("Accuracy:", acc)

# if acc < 0.55:
#     st.error("⚠️ Drift detected!")
# else:
#     st.success("✅ Model stable")

import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("📈 Stock Drift Detection System")

ticker = st.text_input("Enter Stock Symbol", "AAPL")

data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# Indicators
data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Return'] > 0).astype(int)

data['MA5'] = data['Close'].rolling(5).mean()
data['MA10'] = data['Close'].rolling(10).mean()

# RSI
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

data = data.dropna()

if data.empty:
    st.error("No data available. Try another stock.")
    st.stop()

# Load model
model = joblib.load("model.pkl")

X = data[['MA5', 'MA10']]
y = data['Target']

if data.empty:
    st.error("No data available. Try another stock.")
    st.stop()

preds = model.predict(X)
acc = (preds == y).mean()

st.write("### 📊 Accuracy:", acc)

# Drift check
if acc < 0.55:
    st.error("⚠️ Model Drift Detected!")
else:
    st.success("✅ Model Stable")

# Plot graph
st.write("### 📉 Stock Price")
st.line_chart(data['Close'])