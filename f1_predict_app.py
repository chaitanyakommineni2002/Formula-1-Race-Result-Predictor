import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("The trained model file 'trained_model.pkl' is missing. Please upload it to the same directory as this app.")
    st.stop()

# Streamlit app
st.title("Formula 1 Race Result Predictor")
st.write("""
This app predicts whether a driver will finish on the podium based on various factors like driver experience, constructor performance, and grid position.
""")

# Sidebar inputs
st.sidebar.header("Input Race Data")
driver_experience = st.sidebar.slider("Driver Experience (No. of Races)", 0, 300, 50)
constructor_experience = st.sidebar.slider("Constructor Experience (No. of Races)", 0, 300, 50)
grid = st.sidebar.slider("Grid Position", 1, 20, 10)
laps = st.sidebar.slider("Number of Laps Completed", 1, 70, 50)
fastest_lap_speed = st.sidebar.slider("Fastest Lap Speed (km/h)", 150, 400, 250)

# Collect inputs
race_data = pd.DataFrame({
    'driver_experience': [driver_experience],
    'constructor_experience': [constructor_experience],
    'grid': [grid],
    'laps': [laps],
    'fastestLapSpeed': [fastest_lap_speed]
})

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(race_data)
        result = "Podium Finish" if prediction[0] == 1 else "Non-Podium Finish"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# File upload for batch predictions
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])
if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)
        batch_prediction = model.predict(batch_data)
        batch_data['Prediction'] = ["Podium" if p == 1 else "Non-Podium" for p in batch_prediction]
        st.write(batch_data)
        st.download_button("Download Predictions", batch_data.to_csv(index=False), file_name="predictions.csv")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
