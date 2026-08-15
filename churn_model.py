"""
churn_model.py
---------------
STEP 3 of the pipeline: train a Random Forest churn classifier, handling
class imbalance with SMOTE, and evaluate with stratified k-fold cross-
validated ROC-AUC — the "feature engineering, SMOTE, Random Forest, hitting
X% ROC-AUC over Y-fold CV" part of the project.

Key design choices (worth knowing for an interview walkthrough):
  - SMOTE is applied INSIDE the CV loop (via imblearn's Pipeline), not on
    the full dataset before splitting. Applying SMOTE before splitting
    leaks synthetic neighbors of test-set points into training, inflating
    ROC-AUC artificially — a very common mistake in tutorial code.
  - ROC-AUC (not accuracy) is the headline metric because churn is
    imbalanced (~26% positive class here) — a model that just predicts
    "no churn" for everyone would still get ~74% accuracy while being
    useless, whereas ROC-AUC reflects ranking quality regardless of the
    class ratio.
  - Random Forest is used both for solid tabular baseline performance and
    because feature_importances_ gives an interpretable "what's driving
    predictions" output — important for a retention team to trust and act on.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (
    FEATURES_CSV, MODEL_PATH, MODEL_METRICS_JSON,
    RANDOM_STATE, N_CV_FOLDS, TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_features():
    """Load the engineered feature set and split into X (features) / y (target)."""
    df = pd.read_csv(FEATURES_CSV)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def build_pipeline() -> ImbPipeline:
    """
    SMOTE + RandomForest wrapped in an imblearn Pipeline so that SMOTE is
    correctly re-fit on ONLY the training fold during cross-validation
    (never touching validation/test data) — this is what makes the
    reported ROC-AUC trustworthy rather than optimistic.
    """
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight=None,       # SMOTE already handles imbalance; avoid double-correcting
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def cross_validate_model(X: pd.DataFrame, y: pd.Series, n_folds: int = N_CV_FOLDS) -> dict:
    """
    Stratified K-fold CV (stratified so each fold keeps the same ~26%
    churn rate as the full dataset) reporting ROC-AUC per fold plus the
    mean/std — the standard way to report model performance robustly
    rather than relying on a single lucky/unlucky train-test split.
    """
    pipeline = build_pipeline()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

    return {
        "n_folds": n_folds,
        "fold_scores": [round(s, 4) for s in scores],
        "mean_roc_auc": round(scores.mean(), 4),
        "std_roc_auc": round(scores.std(), 4),
    }


def train_final_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train on a held-out train/test split (separate from the CV loop above)
    to report a final confusion matrix and classification report — the
    kind of concrete "how many at-risk customers would we actually catch"
    numbers a retention team needs, not just an abstract AUC score.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    test_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importance from the fitted Random Forest — this is what
    # justifies "prioritizing Z fixes" in the resume bullet: it's a ranked,
    # data-backed list of what actually predicts churn, not a guess.
    feature_importance = pd.Series(
        pipeline.named_steps["clf"].feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    joblib.dump(pipeline, MODEL_PATH)

    return {
        "test_set_roc_auc": round(test_auc, 4),
        "confusion_matrix": {"labels": ["No Churn", "Churn"], "matrix": cm},
        "precision_recall_f1": {
            "no_churn": {k: round(v, 3) for k, v in report["0"].items() if k != "support"},
            "churn": {k: round(v, 3) for k, v in report["1"].items() if k != "support"},
        },
        "top_10_features": feature_importance.head(10).round(4).to_dict(),
    }


def run_modeling():
    X, y = load_features()
    logger.info(f"Loaded {len(X)} rows, {X.shape[1]} features. "
                f"Churn rate: {y.mean() * 100:.1f}%")

    logger.info(f"Running {N_CV_FOLDS}-fold stratified cross-validation with SMOTE + Random Forest...")
    cv_results = cross_validate_model(X, y)

    logger.info("Training final model on train/test split for detailed metrics...")
    final_results = train_final_model(X, y)

    all_metrics = {"cross_validation": cv_results, "held_out_test_set": final_results}
    with open(MODEL_METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n=== {N_CV_FOLDS}-Fold Cross-Validated ROC-AUC ===")
    print(f"Fold scores: {cv_results['fold_scores']}")
    print(f"Mean ROC-AUC: {cv_results['mean_roc_auc']} (+/- {cv_results['std_roc_auc']})")

    print(f"\n=== Held-out Test Set ===")
    print(f"Test ROC-AUC: {final_results['test_set_roc_auc']}")
    print(f"Confusion matrix {final_results['confusion_matrix']['labels']}:")
    for row in final_results["confusion_matrix"]["matrix"]:
        print(f"  {row}")
    print(f"Churn class -> precision: {final_results['precision_recall_f1']['churn']['precision']}, "
          f"recall: {final_results['precision_recall_f1']['churn']['recall']}, "
          f"f1: {final_results['precision_recall_f1']['churn']['f1-score']}")

    print(f"\n=== Top 10 churn drivers (Random Forest feature importance) ===")
    for feat, importance in final_results["top_10_features"].items():
        print(f"  {feat}: {importance}")

    return all_metrics


if __name__ == "__main__":
    run_modeling()
