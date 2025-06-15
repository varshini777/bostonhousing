import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Streamlit App
st.title("🏠 House Price Prediction")

st.markdown("### Enter the feature values to predict house price")

# Form inputs (Replace with actual feature names if needed)
# You can add or rename features here based on your model
feature_1 = st.number_input("Feature 1", format="%.4f")
feature_2 = st.number_input("Feature 2", format="%.4f")
feature_3 = st.number_input("Feature 3", format="%.4f")
feature_4 = st.number_input("Feature 4", format="%.4f")
feature_5 = st.number_input("Feature 5", format="%.4f")
feature_6 = st.number_input("Feature 6", format="%.4f")
feature_7 = st.number_input("Feature 7", format="%.4f")
feature_8 = st.number_input("Feature 8", format="%.4f")
feature_9 = st.number_input("Feature 9", format="%.4f")
feature_10 = st.number_input("Feature 10", format="%.4f")
feature_11 = st.number_input("Feature 11", format="%.4f")
feature_12 = st.number_input("Feature 12", format="%.4f")
feature_13 = st.number_input("Feature 13", format="%.4f")

# Predict button
if st.button("Predict"):
    data = [feature_1, feature_2, feature_3, feature_4, feature_5,
            feature_6, feature_7, feature_8, feature_9, feature_10,
            feature_11, feature_12, feature_13]

    # Mimic Flask debug prints
    st.write("Input Data (Raw):", data)
    array_data = np.array(data).reshape(1, -1)
    st.write("Input Data (Array):", array_data)

    new_data = scalar.transform(array_data)
    prediction = regmodel.predict(new_data)[0]
    
    st.success(f"✅ The House Price Prediction is: **${prediction:,.2f}**")
