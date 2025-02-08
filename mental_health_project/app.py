import streamlit as st
import pandas as pd
import joblib
import os

# Ensure model exists before loading
if not os.path.exists("mental_health_model.pkl"):
    st.error("🚨 ERROR: 'mental_health_model.pkl' not found! Run 'train_model.py' first.")
    st.stop()

# Load Model & Feature Names
loaded_data = joblib.load("mental_health_model.pkl")

# Ensure correct unpacking
if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
    model, feature_names = loaded_data
else:
    st.error("🚨 ERROR: Unexpected data format in 'mental_health_model.pkl'. Expected (model, feature_names).")
    st.stop()

# Function to predict mental health condition
def predict(symptoms):
    """Predicts mental health condition based on user input."""
    # Convert user input into DataFrame
    input_data = pd.DataFrame([symptoms], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_data)[0]

    return prediction

# Streamlit UI
st.title("🧠 Self-Analysis Mental Health Model")

# User Input Fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
work_interfere = st.selectbox("Work Interference", ["No", "Sometimes", "Often", "Always"])
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])
treatment = st.selectbox("Have you sought treatment before?", ["No", "Yes"])

# Mapping categorical inputs to numerical values
gender_map = {"Male": 1, "Female": 0, "Other": 0}
work_map = {"No": 0, "Sometimes": 1, "Often": 2, "Always": 3}
family_map = {"No": 0, "Yes": 1}
self_employed_map = {"No": 0, "Yes": 1}
treatment_map = {"No": 0, "Yes": 1}

# Prepare input for prediction
user_input = {
    "age": age,
    "Gender_Male": gender_map[gender],
    "work_interfere": work_map[work_interfere],
    "family_history": family_map[family_history],
    "self_employed": self_employed_map[self_employed],
    "treatment": treatment_map[treatment]
}

# Ensure input columns match model feature names
user_input_df = pd.DataFrame([user_input])
missing_columns = set(feature_names) - set(user_input_df.columns)

# Handle missing columns
for col in missing_columns:
    user_input[col] = 0  # Assign default value

# Make Prediction
if st.button("Predict 🏥"):
    prediction = predict(user_input)
    st.success(f"🩺 Predicted Condition: {prediction}")
