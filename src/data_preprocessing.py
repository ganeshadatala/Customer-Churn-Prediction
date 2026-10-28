import pandas as pd


def load_data(file_path):
    """Load the customer churn dataset."""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """Clean the customer churn dataset."""

    # Convert TotalCharges into numeric values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Remove customerID because it is an identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # Remove rows where the target is missing
    df = df.dropna(subset=["Churn"])

    return df

