# Power BI Dashboard Build Guide

This is the one part of the pipeline that has to be built by hand — Power BI
Desktop has no CLI or API for programmatic report creation, so the CSVs
below need to be dragged onto a canvas manually. Everything is pre-
aggregated so this should take under an hour.

## 1. Import the data

Open Power BI Desktop → **Get Data → Text/CSV** → import all four files from
`data/powerbi_exports/`:

| File | Grain | Use for |
|---|---|---|
| `churn_by_segment.csv` | one row per segment value | Churn driver bar charts |
| `service_adoption_funnel.csv` | one row per # add-on services (0-6) | Funnel visual |
| `cohort_retention.csv` | one row per tenure cohort | Retention curve |
| `cltv_by_customer.csv` | one row per customer | CLTV distribution, risk-tier table |

## 2. Suggested pages & visuals

**Page 1 — Churn Overview**
- KPI cards: total customers, overall churn rate, MRR at risk (sum of
  `MonthlyCharges` where `risk_tier = "High Risk"` from `cltv_by_customer.csv`)
- Bar chart: `churn_rate_pct` by `segment_value`, sliced by `segment_type`
  (from `churn_by_segment.csv`) — this is your "churn drivers" visual
- Donut chart: customer count by `risk_tier`

**Page 2 — Retention & Funnel**
- Line/column chart: `churn_rate_pct` by `tenure_cohort` (from
  `cohort_retention.csv`) — the retention curve
- Funnel visual: `customer_count` by `num_addon_services`, with
  `churn_rate_pct` as a data label (from `service_adoption_funnel.csv`)

**Page 3 — CLTV & Revenue at Risk**
- Scatter plot: `predicted_cltv` (Y) vs `churn_probability` (X), colored by
  `risk_tier`, from `cltv_by_customer.csv` — instantly shows which
  customers are both high-value AND high-risk (the ones worth a retention
  call)
- Table: top 20 customers by `predicted_cltv` where `risk_tier = "High Risk"`
  — this is the literal "flag at-risk customers" deliverable a retention
  team would work off

## 3. DAX measures worth adding

```dax
Total Revenue At Risk =
CALCULATE(SUM(cltv_by_customer[MonthlyCharges]), cltv_by_customer[risk_tier] = "High Risk")

Overall Churn Rate =
DIVIDE(
    CALCULATE(COUNTROWS(cltv_by_customer), cltv_by_customer[Churn] = "Yes"),
    COUNTROWS(cltv_by_customer)
)

Avg CLTV (High Risk) =
CALCULATE(AVERAGE(cltv_by_customer[predicted_cltv]), cltv_by_customer[risk_tier] = "High Risk")
```

## 4. Publishing

**File → Publish → Publish to Power BI** (requires a free Power BI account)
gets you a shareable web link — useful for the resume/portfolio if you want
to link a live dashboard rather than just screenshots.
