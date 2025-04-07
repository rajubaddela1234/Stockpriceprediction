import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# Page configuration
st.set_page_config(
    page_title="TCS Stock Predictor",
    page_icon="📈",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model_cache():
    return load_model("best_lstm_model.h5", compile=False)

model = load_model_cache()

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('TCS_no_header.csv', skiprows=2)
    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Prediction function
def predict_future_stocks(data, days_to_predict):
    last_100_days = data['Close'][-100:].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(last_100_days)

    predictions = []
    for _ in range(days_to_predict):
        input_data = scaled_data[-100:].reshape(1, 100, 1)
        prediction = model.predict(input_data, verbose=0)[0][0]
        predictions.append(prediction)
        scaled_data = np.append(scaled_data, [[prediction]], axis=0)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = data.index[-1]
    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days_to_predict)]
    
    predicted_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price (₹)': predicted_prices.flatten()
    }).set_index('Date')
    
    # Calculate daily percentage change
    predicted_df['Daily Change (%)'] = predicted_df['Predicted Price (₹)'].pct_change() * 100
    predicted_df['Daily Change (%)'].iloc[0] = ((predicted_df['Predicted Price (₹)'].iloc[0] - data['Close'].iloc[-1]) / 
                                       data['Close'].iloc[-1] * 100)
    
    # Calculate cumulative change from current price
    predicted_df['Cumulative Change (%)'] = ((predicted_df['Predicted Price (₹)'] - current_price) / current_price) * 100
    
    return predicted_df

# UI Components
st.title("📈 TCS Stock Predictor")

data = load_data()
current_price = data['Close'].iloc[-1]
last_date = data.index[-1].strftime('%b %d, %Y')

st.subheader(f"Current Price: ₹{current_price:.2f} (as of {last_date})")

# User input
days = st.number_input(
    "Enter number of days to predict:", 
    min_value=1,
    value=7,
    step=1,
    format="%d"
)

if st.button("Generate Predictions"):
    with st.spinner('Calculating predictions...'):
        predictions = predict_future_stocks(data, days)
        
        # Calculate overall change
        predicted_price = predictions['Predicted Price (₹)'].iloc[-1]
        overall_change = ((predicted_price - current_price) / current_price) * 100
        
        # Show summary metrics
        st.subheader("Prediction Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"₹{current_price:.2f}")
        with col2:
            st.metric(f"Day {days} Price", f"₹{predicted_price:.2f}")
        with col3:
            st.metric("Total Change", 
                     f"{overall_change:.2f}%",
                     delta_color="normal")
        
        # Detailed predictions table
        st.subheader("Daily Price Predictions")
        
        # Format and display the table
        formatted_df = predictions.copy()
        formatted_df.index = formatted_df.index.strftime('%Y-%m-%d')
        
        st.dataframe(
            formatted_df.style.format({
                'Predicted Price (₹)': '₹{:,.2f}',
                'Daily Change (%)': '{:+.2f}%',
                'Cumulative Change (%)': '{:+.2f}%'
            }).applymap(
                lambda x: 'color: green' if x >= 0 else 'color: red',
                subset=['Daily Change (%)', 'Cumulative Change (%)']
            ),
            height=min(800, 35 * len(predictions) + 35),
            width=1000,
            use_container_width=True
        )
        
        # Add download button
        csv = predictions.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f'tcs_predictions_{days}_days.csv',
            mime='text/csv'
        )