import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the trained model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the function to predict power consumption
def predict_power_consumption(temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows, month, day, hour, minute):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'WindSpeed': [wind_speed],
        'GeneralDiffuseFlows': [general_diffuse_flows],
        'DiffuseFlows': [diffuse_flows],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'Minute': [minute]
    })
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Use the model to predict power consumption
    predicted_power_consumption = rf_model.predict(input_data_scaled)
    return predicted_power_consumption[0]

# Design the web app
def main():
    st.title('Power Consumption Prediction')

    # Create input fields for user input
    temperature = st.number_input('Temperature (In Degree Celsius)', min_value=-0.0, max_value=50.0, value=20.0)
    humidity = st.number_input('Humidity (In Percentage In Air )', min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input('Wind Speed (in meter/second)', min_value=0.0, max_value=50.0, value=10.0)
    general_diffuse_flows = st.number_input('General Diffuse Flows ()', min_value=0.0, max_value=1.0, value=0.5)
    diffuse_flows = st.number_input('Diffuse Flows', min_value=0.0, max_value=1.0, value=0.5)
    month = st.number_input('Month(In number 1 to 12)', min_value=1, max_value=12, value=1)
    day = st.number_input('Day (In number 1 to 31)', min_value=1, max_value=31, value=1)
    hour = st.number_input('Hour (In number 0 to 23)', min_value=0, max_value=23, value=0)
    minute = st.number_input('Minute (In Number 0 to 59)', min_value=0, max_value=59, value=0)

    if st.button('Predict'):
        # Make prediction when the button is clicked
        predicted_power = predict_power_consumption(temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows, month, day, hour, minute)
        st.write(f'Predicted Average Power Consumption: {predicted_power} W/h for the next 10 minutes from provided time <br> Provided Time {hour}:{minute} and date {day}/{month}    ')

if __name__ == "__main__":
    main()
