"""
run_pipeline.py
----------------
Orchestrator: runs the full churn prediction pipeline end-to-end.

    1. SQL analysis          (sql_analysis.py)          -> churn drivers, funnel, cohorts
    2. Feature engineering   (feature_engineering.py)    -> model-ready feature matrix
    3. Model training        (churn_model.py)            -> SMOTE + Random Forest + CV
    4. CLTV + Power BI export (cltv_and_powerbi_export.py) -> dashboard-ready CSVs

Usage:
    python run_pipeline.py
"""

import logging

import sql_analysis
import feature_engineering
import churn_model
import cltv_and_powerbi_export

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("STEP 1/4: SQL analysis (churn drivers, funnel, cohorts)...")
    sql_analysis.run_sql_analysis()

    logger.info("STEP 2/4: Feature engineering...")
    features_df = feature_engineering.build_feature_set()
    features_df.to_csv(feature_engineering.FEATURES_CSV, index=False)

    logger.info("STEP 3/4: Training SMOTE + Random Forest churn model...")
    churn_model.run_modeling()

    logger.info("STEP 4/4: Computing CLTV and building Power BI exports...")
    cltv_and_powerbi_export.run_cltv_and_export()

    logger.info(
        "Pipeline complete. All Power BI-ready CSVs are in data/powerbi_exports/. "
        "See POWERBI_BUILD_GUIDE.md for how to assemble the dashboard."
    )


if __name__ == "__main__":
    main()
