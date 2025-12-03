import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(page_title="Youth Unemployment Predictor", page_icon="📈")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'youth_unemployment_global.csv')
model_path = os.path.join(current_dir, 'unemployment_model.joblib')
encoder_path = os.path.join(current_dir, 'country_encoder.joblib')

# Load resources
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_model_resources():
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

try:
    df = load_data()
    model, encoder = load_model_resources()
except FileNotFoundError:
    st.error("Model or Data files not found. Please run train_model.py first.")
    st.stop()

# App Title and Description
st.title("📈 Youth Unemployment Rate Predictor")
st.markdown("""
This app predicts the **Youth Unemployment Rate** for a specific country and year using a Random Forest model.
Select a country and year below to see the prediction and historical trends.
""")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Country Selection
countries = sorted(df['Country'].unique())
selected_country = st.sidebar.selectbox("Select Country", countries)

# Year Selection
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
selected_year = st.sidebar.slider("Select Year", min_year, max_year + 10, max_year + 1)

# Prediction Logic
if st.sidebar.button("Predict"):
    # Encode country
    try:
        country_encoded = encoder.transform([selected_country])[0]
        
        # Make prediction
        prediction = model.predict([[country_encoded, selected_year]])[0]
        
        st.success(f"Predicted Youth Unemployment Rate for **{selected_country}** in **{selected_year}**: **{prediction:.2f}%**")
        
        # --- Visualization ---
        st.subheader(f"Historical Trend & Prediction for {selected_country}")
        
        # Filter data for the selected country
        country_data = df[df['Country'] == selected_country].sort_values('Year')
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        
        # Historical Line
        sns.lineplot(x='Year', y='YouthUnemployment', data=country_data, ax=ax, label='Historical Data', marker='o')
        
        # Predicted Point
        ax.scatter(selected_year, prediction, color='red', s=100, label='Prediction', zorder=5)
        
        ax.set_title(f"Youth Unemployment Trend: {selected_country}")
        ax.set_ylabel("Unemployment Rate (%)")
        ax.set_xlabel("Year")
        ax.legend()
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Show raw data option
if st.checkbox("Show Raw Data for Selected Country"):
    st.write(df[df['Country'] == selected_country])
