import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# Make sure 'random_forest_regressor_model.pkl' is in the same directory
with open('random_forest_regressor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the original dataset to fit LabelEncoders
# Make sure 'india_rainfall_data.csv' is in the same directory
df_original = pd.read_csv('india_rainfall_data.csv')

# Initialize and fit LabelEncoders for STATE_UT_NAME and DISTRICT
le_state = LabelEncoder()
le_district = LabelEncoder()

le_state.fit(df_original['STATE_UT_NAME'])
le_district.fit(df_original['DISTRICT'])

# Streamlit app title
st.title('Annual Rainfall Prediction App')

st.write('Select a State/UT and District to predict the annual rainfall.')

# Create dropdowns for State and District
selected_state_name = st.selectbox(
    'Select State/UT Name:',
    df_original['STATE_UT_NAME'].unique()
)

# Filter districts based on selected state
available_districts = df_original[df_original['STATE_UT_NAME'] == selected_state_name]['DISTRICT'].unique()
selected_district_name = st.selectbox(
    'Select District:',
    available_districts
)

if st.button('Predict Annual Rainfall'):
    try:
        # Encode the selected state and district
        encoded_state = le_state.transform([selected_state_name])[0]
        encoded_district = le_district.transform([selected_district_name])[0]

        # Create a DataFrame for prediction (matching the training features)
        input_data = pd.DataFrame([[encoded_state, encoded_district]], 
                                  columns=['STATE_UT_ENCODED', 'DISTRICT_ENCODED'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.success(f'Predicted Annual Rainfall: {prediction:.2f} mm')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
