import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from dill import loads
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main():
    st.title('FIFA Player Rating Predictor')

    # Video link at the top
    st.video('https://github.com/kelvin-ahiakpor/KELVINAHIAKPOR._SportsPrediction8/blob/main/Kelvin%20Fifa%20Regression%20DEMO.mp4')

    # Load model and scaler
    model = load_model()
    scaler = load_scaler()

    # Sidebar with input fields for features
    movement_reactions = st.number_input('Enter Movement Reactions', min_value=0, step=1)
    potential = st.number_input('Enter Potential', min_value=0, step=1)
    wage_eur = st.number_input('Enter Wage (EUR)', min_value=0, step=1)
    release_clause_eur = st.number_input('Enter Release Clause (EUR)', min_value=0, step=1)
    value_eur = st.number_input('Enter Value (EUR)', min_value=0, step=1)
    passing = st.number_input('Enter Passing', min_value=0, step=1)
    dribbling = st.number_input('Enter Dribbling', min_value=0, step=1)

    # Prepare player data for prediction
    player_data = {
        'movement_reactions': movement_reactions,
        'potential': potential,
        'wage_eur': wage_eur,
        'release_clause_eur': release_clause_eur,
        'value_eur': value_eur,
        'passing': passing,
        'dribbling': dribbling
    }

    # Predict button
    if st.button('Predict Player Rating'):
        # Make prediction
        predicted_rating, lower_bound, upper_bound = predict_overall(model, scaler, player_data)

        # Display results
        st.success(f'Predicted Overall Rating: {predicted_rating:.2f}')
        st.info(f'Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]')

# Running the app
if __name__ == '__main__':
    main()
