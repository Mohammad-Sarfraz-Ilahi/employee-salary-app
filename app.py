import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('knn_salary_model.pkl')
encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction using KNN")

st.markdown("Predict whether an employee earns **more or less than 50K** based on personal and work-related information.")

# Sidebar form
st.sidebar.header("🧾 Enter Employee Details")

def user_input():
    age = st.sidebar.slider('Age', 18, 90, 30)
    workclass = st.sidebar.selectbox('Workclass', encoders['workclass'].classes_)
    education = st.sidebar.selectbox('Education', encoders['education'].classes_)
    marital_status = st.sidebar.selectbox('Marital Status', encoders['marital-status'].classes_)
    occupation = st.sidebar.selectbox('Occupation', encoders['occupation'].classes_)
    relationship = st.sidebar.selectbox('Relationship', encoders['relationship'].classes_)
    race = st.sidebar.selectbox('Race', encoders['race'].classes_)
    gender = st.sidebar.selectbox('Gender', encoders['gender'].classes_)
    capital_gain = st.sidebar.number_input('Capital Gain', min_value=0, value=0)
    capital_loss = st.sidebar.number_input('Capital Loss', min_value=0, value=0)
    hours_per_week = st.sidebar.slider('Hours per Week', 1, 100, 40)
    native_country = st.sidebar.selectbox('Native Country', encoders['native-country'].classes_)

    input_data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    return pd.DataFrame([input_data])

# Collect user input
input_df = user_input()

# Encode input using saved encoders
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Show entered data
st.subheader("📋 Entered Data")
st.write(input_df)

# Predict and show result
if st.button("🔮 Predict Salary Category"):
    prediction = model.predict(input_df)
    result = target_encoder.inverse_transform(prediction)[0]
    st.success(f"💰 **Predicted Salary:** {result}")
