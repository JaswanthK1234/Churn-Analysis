"""
config.py
---------
Central settings for the churn prediction project.

DATA SOURCE: This project ships with the IBM Telco Customer Churn dataset
(7,043 customers, 21 columns) as a realistic stand-in for a real company's
customer database. It's the industry-standard public dataset for churn
portfolio projects. Swap `RAW_DATA_CSV` for your own export (same schema:
one row per customer, a binary Churn column, tenure + service + billing
fields) to run this on real company data instead.
"""

import os

DATA_DIR = "data"
RAW_DATA_CSV = f"{DATA_DIR}/Telco-Customer-Churn.csv"
SQLITE_DB_PATH = f"{DATA_DIR}/churn.db"

FEATURES_CSV = f"{DATA_DIR}/engineered_features.csv"
MODEL_PATH = f"{DATA_DIR}/churn_model.joblib"
MODEL_METRICS_JSON = f"{DATA_DIR}/model_metrics.json"

# Power BI-ready export files (aggregated, no row-level PII)
POWERBI_DIR = f"{DATA_DIR}/powerbi_exports"
CHURN_BY_SEGMENT_CSV = f"{POWERBI_DIR}/churn_by_segment.csv"
COHORT_RETENTION_CSV = f"{POWERBI_DIR}/cohort_retention.csv"
FUNNEL_CSV = f"{POWERBI_DIR}/service_adoption_funnel.csv"
CLTV_CSV = f"{POWERBI_DIR}/cltv_by_customer.csv"

# Modeling settings
RANDOM_STATE = 42
N_CV_FOLDS = 5
TEST_SIZE = 0.2

# Business assumption for CLTV: average gross margin on monthly revenue.
# Adjust to your company's real unit economics.
GROSS_MARGIN = 0.60

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(POWERBI_DIR, exist_ok=True)
