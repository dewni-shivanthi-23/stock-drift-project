# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt

# st.title("Stock Drift Detection System")

# ticker = st.text_input("Enter Stock Symbol", "AAPL")

# data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# # Indicators
# data['Return'] = data['Close'].pct_change()
# data['Target'] = (data['Return'] > 0).astype(int)

# data['MA5'] = data['Close'].rolling(5).mean()
# data['MA10'] = data['Close'].rolling(10).mean()

# # RSI
# delta = data['Close'].diff()
# gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# rs = gain / loss
# data['RSI'] = 100 - (100 / (1 + rs))

# data = data.dropna()

# if data.empty:
#     st.error("No data available. Try another stock.")
#     st.stop()

# # Load model
# model = joblib.load("model.pkl")

# X = data[['MA5', 'MA10']]
# y = data['Target']

# if data.empty:
#     st.error("No data available. Try another stock.")
#     st.stop()

# preds = model.predict(X)
# acc = (preds == y).mean()

# st.write("### Accuracy:", acc)

# # Drift check
# if acc < 0.55:
#     st.error(" Model Drift Detected!")
# else:
#     st.success(" Model Stable")

# # Plot graph
# st.write("### Stock Price")
# st.line_chart(data['Close'])

import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

st.title("Multi-Stock Drift Detection Dashboard")

# Multi-stock input
stocks = st.multiselect(
    "Select Stocks",
    ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN"],
    default=["AAPL", "TSLA"]
)

# Load model
model = joblib.load("model.pkl")

for ticker in stocks:
    st.subheader(f" {ticker}")

    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

    # Features
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Return'] > 0).astype(int)

    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()

    data = data.dropna()

    if data.empty:
        st.warning("No data available")
        continue

    X = data[['MA5', 'MA10']]
    y = data['Target']

    # Safety check
    if X.isnull().values.any():
        st.warning("Missing data")
        continue

    preds = model.predict(X)
    acc = (preds == y).mean()

    st.write(f"Accuracy: {round(acc, 2)}")

    if acc < 0.55:
        st.error(" Drift Detected!")
    else:
        st.success(" Model Stable")

    # Graph
    st.line_chart(data['Close'])