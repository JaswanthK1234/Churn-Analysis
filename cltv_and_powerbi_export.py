"""
cltv_and_powerbi_export.py
----------------------------
STEP 4 of the pipeline: compute Customer Lifetime Value (CLTV) and shape
every output the dashboard needs into clean, pre-aggregated CSVs that
Power BI can import directly — the data layer behind "Designed Power BI
dashboards quantifying churn, retention & CLTV."

IMPORTANT — what this script does NOT do: it does not (and cannot) build
the actual .pbix Power BI file. Power BI Desktop is proprietary Windows/
Mac software with no API or CLI for programmatic report building, so the
dashboard itself has to be assembled by hand in Power BI Desktop. What
this script DOES do is remove all the tedious data-shaping work so that
step is closer to just "drag fields onto a canvas" — see
POWERBI_BUILD_GUIDE.md for the exact visuals + DAX measures to add.
"""

import logging
import pandas as pd

from config import (
    RAW_DATA_CSV, MODEL_PATH, FEATURES_CSV,
    CLTV_CSV, GROSS_MARGIN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_cltv(df: pd.DataFrame, churn_probability: pd.Series) -> pd.DataFrame:
    """
    Simple, standard subscription CLTV formula:

        CLTV = (Monthly Charges x Gross Margin) / Monthly Churn Probability

    This treats each customer's predicted churn probability as their
    personal "monthly attrition rate," so a customer predicted at 5% churn
    risk gets a much longer expected lifetime (and higher CLTV) than one
    predicted at 40% risk. Clamping churn probability to a small floor
    avoids division blowing up for near-zero-risk customers.
    """
    out = df[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()
    out["churn_probability"] = churn_probability.values

    floor = 0.02  # avoid divide-by-near-zero for very low-risk customers
    safe_prob = out["churn_probability"].clip(lower=floor)

    out["predicted_cltv"] = (out["MonthlyCharges"] * GROSS_MARGIN) / safe_prob
    out["historical_value_to_date"] = out["TotalCharges"]

    # Simple risk tier for dashboard slicers / conditional formatting
    out["risk_tier"] = pd.cut(
        out["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    return out.round(2)


def run_cltv_and_export():
    import joblib

    logger.info("Loading trained model and features to score churn probability for every customer...")
    model = joblib.load(MODEL_PATH)
    features_df = pd.read_csv(FEATURES_CSV)
    raw_df = pd.read_csv(RAW_DATA_CSV)

    X = features_df.drop(columns=["Churn"])
    churn_probability = pd.Series(model.predict_proba(X)[:, 1], index=raw_df.index)

    cltv_df = compute_cltv(raw_df, churn_probability)
    cltv_df.to_csv(CLTV_CSV, index=False)

    logger.info(f"Saved per-customer CLTV + risk scores for {len(cltv_df)} customers -> {CLTV_CSV}")

    print("\n=== CLTV summary by risk tier ===")
    summary = cltv_df.groupby("risk_tier", observed=True).agg(
        customer_count=("customerID", "count"),
        avg_predicted_cltv=("predicted_cltv", "mean"),
        total_revenue_at_risk=("MonthlyCharges", "sum"),
    ).round(2)
    print(summary.to_string())

    high_risk_revenue = cltv_df.loc[cltv_df["risk_tier"] == "High Risk", "MonthlyCharges"].sum()
    print(f"\nMonthly recurring revenue sitting in the High Risk tier: ${high_risk_revenue:,.2f}"
          f"  (this is the number a retention team would size an intervention budget against)")

    return cltv_df


if __name__ == "__main__":
    run_cltv_and_export()
