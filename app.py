import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# Find the saved model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Application title
st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether "
    "the customer may leave the service."
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    pipeline = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. "
        "Please train the model using src/train_model.py."
    )
    st.stop()


# Customer details form
st.header("Customer Information")

with st.form("customer_form"):

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Has Partner?",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    submitted = st.form_submit_button(
        "Predict Churn"
    )


# Make prediction
if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    customer_df = pd.DataFrame([customer_data])

    prediction = pipeline.predict(customer_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer may churn.")
    else:
        st.success("✅ Customer may not churn.")

    # Show estimated churn probability
    probability = pipeline.predict_proba(customer_df)[0][1]

    st.metric(
        label="Estimated Churn Probability",
        value=f"{probability:.2%}"
    )

    st.caption(
        "This is a machine learning estimate, "
        "not a guarantee of customer behavior."
    )
