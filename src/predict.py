import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"


def predict_churn(customer_data):
    """Predict whether a customer may leave."""

    pipeline = joblib.load(MODEL_PATH)

    customer_df = pd.DataFrame([customer_data])

    prediction = pipeline.predict(customer_df)[0]

    if prediction == 1:
        return "Customer may churn"
    else:
        return "Customer may not churn"


if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0
    }

    result = predict_churn(customer)
    print("Prediction:", result)

