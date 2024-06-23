import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from dill import loads
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Function to load concatenated parts
def load_parts(prefix, src):
    src = Path(src)
    combined = b""
    parts = []
    for file in src.glob("*.pkl"):
        if file.stem.startswith(prefix):
            parts.append(file)
    parts.sort(key=lambda part: int(part.stem.split("-")[-1]))

    for part in parts:
        with open(part, "rb") as f:
            combined += f.read()

    return loads(combined)

# Load the trained model
def load_model():
    model = load_parts("FifaRandomForestRegressor-", "./models/")
    return model

# Load the scaler
def load_scaler():
    scaler_path = "./scaler.pkl"
    with open(scaler_path, "rb") as scaler_file:
        scaler = pkl.load(scaler_file)
    return scaler

# Function to preprocess input data and predict using the trained model
def predict_overall(model, scaler, player_data):
    # Convert input data into a DataFrame
    player_features = pd.DataFrame(player_data, index=[0])

    # Handle the case where release_clause_eur is zero by setting it to value_eur
    player_features.loc[player_features['release_clause_eur'] == 0, 'release_clause_eur'] = player_features['value_eur']

    # Apply logarithm transformations
    player_features['log_wage_eur'] = np.log(player_features['wage_eur'] + 1)
    player_features['log_value_eur'] = np.log(player_features['value_eur'] + 1)
    player_features['log_release_clause_eur'] = np.log(player_features['release_clause_eur'] + 1)
    player_features['passing_dribbling_interaction'] = player_features['passing'] * player_features['dribbling']

    selected_features = ['log_value_eur', 'log_release_clause_eur', 'movement_reactions',
                         'log_wage_eur', 'potential', 'passing_dribbling_interaction', 'passing', 'wage_eur', 'value_eur']
    player_features = player_features[selected_features]

    # Scale the input data using the loaded scaler
    X_scaled = scaler.transform(player_features)

    # Make predictions
    predicted_rating = model.predict(X_scaled)

    # For demonstration purpose, assuming a fixed margin of error (you can adjust this)
    margin_of_error = 1.96 * 3.18  # Adjust based on your desired confidence level and RMSE

    lower_bound = predicted_rating - margin_of_error
    upper_bound = predicted_rating + margin_of_error

    return predicted_rating[0], lower_bound[0], upper_bound[0]

# Streamlit UI
def main():
    st.title('FIFA Player Rating Predictor')

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