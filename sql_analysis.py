"""
sql_analysis.py
-----------------
STEP 1 of the pipeline: load customer records into a SQL database and run
funnel + cohort analysis to uncover churn drivers — the "Analyzed X+
records via SQL, uncovering Y churn drivers" part of the project.

Uses SQLite so this runs anywhere with no server setup. The SQL itself
(window functions, CASE-based bucketing, GROUP BY aggregations) is the
same style you'd write against Postgres/MySQL/Snowflake in a real company
environment — swap the connection in `get_connection()` for a real DB
driver (e.g. psycopg2, pyodbc) to point this at production data.

Three analyses are run, each answering a different product question:
  1. DRIVER ANALYSIS   -> which customer attributes correlate with churn?
  2. FUNNEL ANALYSIS    -> does adopting more services reduce churn?
  3. COHORT ANALYSIS    -> how does churn risk evolve over customer tenure?
"""

import sqlite3
import logging
import pandas as pd

from config import RAW_DATA_CSV, SQLITE_DB_PATH, CHURN_BY_SEGMENT_CSV, FUNNEL_CSV, COHORT_RETENTION_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database file."""
    return sqlite3.connect(SQLITE_DB_PATH)


def load_csv_into_sql(csv_path: str = RAW_DATA_CSV, table_name: str = "customers") -> int:
    """
    STEP 1a: load the raw customer CSV into a SQL table. In a real company
    this would instead be a scheduled extract from the production database
    or data warehouse (e.g. via an ETL job) — here we simulate that by
    loading a CSV export, which is the most common real-world pattern
    anyway (analysts rarely query production OLTP tables directly).
    """
    df = pd.read_csv(csv_path)

    # TotalCharges has some blank strings for brand-new customers (tenure=0);
    # coerce to numeric and fill with 0 so SQL aggregations don't choke on it.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    conn = get_connection()
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    logger.info(f"Loaded {len(df)} records into SQL table '{table_name}'")
    return len(df)


def churn_driver_analysis(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    STEP 1b (DRIVER ANALYSIS): churn rate broken down by contract type,
    payment method, and internet service — the segments most commonly
    responsible for outsized churn in subscription businesses.

    Uses SQL CASE + GROUP BY, and UNION ALL to stack multiple segment
    breakdowns into one tidy long-format table (easy to drop straight
    into a Power BI table visual with a segment-type slicer).
    """
    query = """
    WITH by_contract AS (
        SELECT
            'Contract Type' AS segment_type,
            Contract AS segment_value,
            COUNT(*) AS customer_count,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
        FROM customers
        GROUP BY Contract
    ),
    by_payment AS (
        SELECT
            'Payment Method' AS segment_type,
            PaymentMethod AS segment_value,
            COUNT(*) AS customer_count,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
        FROM customers
        GROUP BY PaymentMethod
    ),
    by_internet AS (
        SELECT
            'Internet Service' AS segment_type,
            InternetService AS segment_value,
            COUNT(*) AS customer_count,
            ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
        FROM customers
        GROUP BY InternetService
    )
    SELECT * FROM by_contract
    UNION ALL SELECT * FROM by_payment
    UNION ALL SELECT * FROM by_internet
    ORDER BY segment_type, churn_rate_pct DESC;
    """
    return pd.read_sql_query(query, conn)


def service_adoption_funnel(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    STEP 1c (FUNNEL ANALYSIS): does adopting more add-on services
    ("deepening" the relationship) reduce churn? Counts how many of the
    6 optional services (OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies) each customer has, then
    computes churn rate at each adoption depth — a classic product funnel
    read on retention.
    """
    query = """
    WITH service_counts AS (
        SELECT
            customerID,
            Churn,
            (CASE WHEN OnlineSecurity   = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN OnlineBackup     = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN TechSupport      = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN StreamingTV      = 'Yes' THEN 1 ELSE 0 END) +
            (CASE WHEN StreamingMovies  = 'Yes' THEN 1 ELSE 0 END) AS num_addon_services
        FROM customers
    )
    SELECT
        num_addon_services,
        COUNT(*) AS customer_count,
        ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
    FROM service_counts
    GROUP BY num_addon_services
    ORDER BY num_addon_services;
    """
    return pd.read_sql_query(query, conn)


def tenure_cohort_analysis(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    STEP 1d (COHORT ANALYSIS): bucket customers into tenure cohorts
    (0-6 months, 7-12, 13-24, 25-48, 49+) and compute churn rate per
    cohort. This dataset doesn't include signup dates, so tenure-bucket
    cohorts stand in for calendar-month cohorts — the standard approach
    for this dataset and a fair proxy for "how does churn risk change
    as the customer relationship matures?"
    """
    query = """
    WITH cohorts AS (
        SELECT
            customerID,
            Churn,
            MonthlyCharges,
            CASE
                WHEN tenure <= 6  THEN '1. 0-6 mo (new)'
                WHEN tenure <= 12 THEN '2. 7-12 mo'
                WHEN tenure <= 24 THEN '3. 13-24 mo'
                WHEN tenure <= 48 THEN '4. 25-48 mo'
                ELSE '5. 49+ mo (loyal)'
            END AS tenure_cohort
        FROM customers
    )
    SELECT
        tenure_cohort,
        COUNT(*) AS customer_count,
        ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
        ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
    FROM cohorts
    GROUP BY tenure_cohort
    ORDER BY tenure_cohort;
    """
    return pd.read_sql_query(query, conn)


def run_sql_analysis():
    """Load data, run all three analyses, print a summary, and save Power BI-ready exports."""
    n_records = load_csv_into_sql()
    conn = get_connection()

    driver_df = churn_driver_analysis(conn)
    funnel_df = service_adoption_funnel(conn)
    cohort_df = tenure_cohort_analysis(conn)

    conn.close()

    driver_df.to_csv(CHURN_BY_SEGMENT_CSV, index=False)
    funnel_df.to_csv(FUNNEL_CSV, index=False)
    cohort_df.to_csv(COHORT_RETENTION_CSV, index=False)

    print(f"\n=== Analyzed {n_records} customer records via SQL ===")

    print("\n--- Churn drivers by segment (top risk segments) ---")
    print(driver_df.sort_values("churn_rate_pct", ascending=False).head(8).to_string(index=False))

    print("\n--- Service adoption funnel (does more services = less churn?) ---")
    print(funnel_df.to_string(index=False))

    print("\n--- Tenure cohort churn rates ---")
    print(cohort_df.to_string(index=False))

    return driver_df, funnel_df, cohort_df


if __name__ == "__main__":
    run_sql_analysis()
