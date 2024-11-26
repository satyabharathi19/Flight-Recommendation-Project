# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:52:32 2024

@author: bharu
"""

import streamlit as st
import pickle
import pandas as pd

# Load the dataset
df = pd.read_csv('ProcessedDataset.csv')  # Replace with your dataset

# Load the saved models
with open('recommended_model.pkl', 'rb') as file:
    recommended_model = pickle.load(file)

with open('sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

# Streamlit input
input_airline = st.text_input('Enter the flight name:')

# Filter the dataset based on input
if input_airline:
    filtered_flights = df[df['AirName'] == input_airline]

    if not filtered_flights.empty:
        # Extract features for prediction
        features = ['OverallScore', 'EntertainmentRating', 'FoodRating', 'ServiceRating', 'WifiRating']
        input_features = filtered_flights[features]

        # Make predictions
        recommended_pred = recommended_model.predict(input_features)
        sentiment_pred = sentiment_model.predict(input_features)

        # Add predictions to the DataFrame
        filtered_flights['Recommended_pred'] = recommended_pred
        filtered_flights['Sentiment_pred'] = sentiment_pred

        # Filter based on predicted values
        recommended_flights = filtered_flights[
            (filtered_flights['Recommended_pred'] == 1) & 
            (filtered_flights['Sentiment_pred'] == 1)
        ]

        # Display the recommended flights
        st.write(f"Recommended '{input_airline}' flights with good reviews:")
        st.write(recommended_flights[['AirName', 'OverallScore', 'EntertainmentRating', 'ServiceRating', 'Review']])
    else:
        st.write(f"No flights found for '{input_airline}'.")
