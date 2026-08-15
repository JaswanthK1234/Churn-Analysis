"""
feature_engineering.py
------------------------
STEP 2 of the pipeline: turn the raw customer table into a clean,
model-ready feature matrix.

Feature groups created:
  1. Cleaned raw fields       (numeric coercion, binary flags)
  2. Encoded categoricals      (one-hot for contract/payment/internet type etc.)
  3. Engineered features       (num_addon_services, tenure buckets, avg spend
                                 per tenure month) — these came directly out
                                 of the SQL driver/funnel analysis in
                                 sql_analysis.py, which is the point: EDA
                                 findings should feed feature engineering,
                                 not be a disconnected step.
"""

import logging
import pandas as pd

from config import RAW_DATA_CSV, FEATURES_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# The 6 optional add-on services — same list used in the SQL funnel analysis
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Binary Yes/No columns to convert to 1/0
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]

# Multi-category columns to one-hot encode
CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService", "Contract", "PaymentMethod",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_and_clean(csv_path: str = RAW_DATA_CSV) -> pd.DataFrame:
    """Load raw CSV and fix known data-quality issues."""
    df = pd.read_csv(csv_path)

    # TotalCharges is stored as a string and has blanks for tenure=0 customers
    # (they haven't been billed yet) — coerce to numeric, fill with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Drop the ID column — a unique identifier has zero predictive value and
    # including it risks the model latching onto spurious patterns.
    df = df.drop(columns=["customerID"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features discovered to matter during the SQL/EDA phase."""
    df = df.copy()

    # From the funnel analysis: number of add-on services adopted is a
    # strong churn signal (0 services -> higher churn, but so is 1-in-isolation;
    # 5-6 services -> much lower churn). Let the model see this directly
    # rather than making it re-derive it from 6 separate one-hot columns.
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)

    # From the cohort analysis: churn risk is heavily concentrated in the
    # first 6-12 months. Give the model an explicit "new customer" flag,
    # since tree models don't always find sharp thresholds efficiently on
    # their own from a raw continuous tenure column.
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    # Average monthly spend relative to tenure — catches customers who are
    # paying a lot relative to how established they are (price-sensitivity risk)
    df["charges_per_tenure_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Whether the customer has ANY internet service at all (no-internet
    # customers behave very differently — much lower churn, fewer add-ons)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary-encode Yes/No columns and one-hot encode multi-category columns."""
    df = df.copy()

    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df


def build_feature_set() -> pd.DataFrame:
    """Full feature engineering pipeline: load -> clean -> engineer -> encode."""
    df = load_and_clean()
    df = engineer_features(df)
    df = encode_features(df)

    logger.info(f"Built feature set: {df.shape[0]} rows x {df.shape[1]} columns "
                f"({df.shape[1] - 1} features + 1 target)")
    return df


if __name__ == "__main__":
    features_df = build_feature_set()
    features_df.to_csv(FEATURES_CSV, index=False)
    logger.info(f"Saved engineered features -> {FEATURES_CSV}")
    print(features_df.head())
