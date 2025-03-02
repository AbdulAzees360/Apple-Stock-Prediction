import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta

# Load the trained model (Assuming it's a Prophet model)
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv("AAPL.csv")

# Ensure Date column is in datetime format and rename it for Prophet
df["Date"] = pd.to_datetime(df["Date"])
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)  # Prophet requires 'ds' and 'y'

# Predict function
def predict_future_prices(model, df, days=30):
    last_date = df["ds"].max()
    future_dates = pd.DataFrame({"ds": [last_date + timedelta(days=i) for i in range(1, days + 1)]})

    forecast = model.predict(future_dates)
    return forecast[["ds", "yhat"]]  # Prophet outputs 'yhat' for predictions

# Streamlit UI
st.title("📈 Apple Stock Price Prediction")
st.write("This application predicts Apple stock prices for the next 30 days.")

# Display dataset preview
st.subheader("Historical Data")
st.write(df.tail(10))

# Predict future prices
st.subheader("Predicted Prices for Next 30 Days")
future_df = predict_future_prices(model, df)
st.write(future_df)

# Plot results
st.subheader("Stock Price Prediction Chart")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["ds"], df["y"], label="Historical Prices", color="blue")
ax.plot(future_df["ds"], future_df["yhat"], label="Predicted Prices", linestyle="dashed", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)

st.write("🔍 The red dashed line represents the predicted prices for the next 30 days.")

