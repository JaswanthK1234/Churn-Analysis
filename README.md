# Customer Churn Prediction System

End-to-end pipeline that flags at-risk customers, quantifies churn drivers,
and estimates revenue at risk — built to support a retention team's
decisions, not just to produce a model score.

## Pipeline

```
Raw customer data (CSV export)
        |
        v
sql_analysis.py          -> SQLite: churn drivers, service-adoption funnel,
        |                    tenure-cohort retention (real SQL, real answers)
        v
feature_engineering.py   -> cleaned, encoded, engineered feature matrix
        |
        v
churn_model.py            -> SMOTE (inside CV, no leakage) + Random Forest,
        |                     5-fold stratified ROC-AUC, feature importance
        v
cltv_and_powerbi_export.py -> per-customer CLTV + risk tier, Power BI CSVs
        |
        v
Power BI Desktop (manual)  -> dashboard (see POWERBI_BUILD_GUIDE.md)
```

## Setup

```bash
pip install -r requirements.txt
```

No API keys needed — this ships with the public **IBM Telco Customer
Churn** dataset (7,043 customers, `data/Telco-Customer-Churn.csv`), the
standard dataset for churn portfolio projects. To run this on your own
company's data instead, replace that CSV with your own export — same
shape (one row per customer, a binary Churn column, tenure/service/billing
fields) — and adjust the column names in `feature_engineering.py` and
`sql_analysis.py` to match.

## Run

```bash
python run_pipeline.py
```

This runs all 4 Python steps and writes everything to `data/`, including
`data/powerbi_exports/` — the CSVs the Power BI dashboard is built from.
Then follow `POWERBI_BUILD_GUIDE.md` to assemble the dashboard by hand in
Power BI Desktop (no CLI/API exists for this step — it's the one part of
the pipeline you can't script).

## Results on the included dataset (example numbers — yours will differ)

- **7,043 records** analyzed via SQL across 3 driver dimensions
- **Top churn drivers found:** month-to-month contracts (42.7% churn vs.
  11.3% for one-year contracts), electronic check payment (45.3% churn),
  fiber optic internet (41.9% churn) — and adopting 5-6 add-on services
  cuts churn from ~46% down to ~5%
- **83.8% mean ROC-AUC** over 5-fold stratified CV (SMOTE + Random Forest)
- **$136K/month** in recurring revenue sitting in the "High Risk" customer
  tier — the number a retention budget would be sized against

These numbers come from the public dataset included here, not a real
company — swap in your own data before quoting these on a resume.

## Known limitations / where human judgment is still required

- The dataset has no real dates, so "cohort analysis" uses tenure buckets
  as a proxy for calendar-month cohorts — a fair substitute for this
  dataset, but real signup-date cohorts would be more precise on a live
  system.
- The Random Forest here is untuned beyond sane defaults (depth, leaf
  size) — a real deployment would warrant hyperparameter search and
  comparison against gradient boosting (XGBoost/LightGBM) baselines.
- CLTV uses a simplified formula (`monthly revenue x margin / churn
  probability`); a finance team's real CLTV model may include discount
  rates, contract-length effects, or CAC.
- The "cutting churn by X%" business outcome in the resume bullet can only
  be validated after a retention intervention is actually run against
  flagged customers — no dataset can simulate that causal effect.

## Tech stack

Python · SQLite (SQL) · pandas · scikit-learn · imbalanced-learn (SMOTE) ·
Random Forest · Power BI (dashboard, built manually from exports)
