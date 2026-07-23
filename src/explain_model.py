"""
Day 2 - Explain the risk-prediction model's decisions using SHAP.

Purpose:
    A prediction alone ("this review is high-risk") is not trustworthy on
    its own needs to know WHY. SHAP
    breaks each prediction down into the contribution of each individual
    feature, like an itemized receipt.

    This script produces two kinds of explanation:
        1. GLOBAL: across all test reviews, which features matter most
           overall to the model's decisions?
        2. LOCAL: for one specific high-risk review and one specific
           not-high-risk review, exactly which features pushed the
           prediction up or down, and by how much?

Input:
    data/model_features.csv
    models/risk_model.joblib (produced by train_model.py)

Output:
    Printed global feature importance ranking, and two worked local
    examples.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from train_model import DATA_PATH, TARGET_COLUMN, load_features, split_data

MODEL_PATH = Path("models/risk_model.joblib")


def load_trained_model():
    """Load the Logistic Regression model saved by train_model.py."""
    return joblib.load(MODEL_PATH)


def compute_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Compute SHAP values for the test set predictions.

    Different model types need a different SHAP method under the hood:
        - Tree-based models (Random Forest, etc.) use TreeExplainer,
          which reads the trees' structure directly and does not need a
          background dataset.
        - Linear models (Logistic Regression) use LinearExplainer,
          which needs a background dataset to establish the "average"
          starting point (the base value).

    I pick explicitly based on the model's type rather than relying on
    automatic detection, so the logic is easy to explain and reason
    about.
    """
    model_type_name = type(model).__name__

    if model_type_name == "RandomForestClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        # Random Forest explanations include one set of values per class
        # (not-risky, high-risk). We keep only the "high-risk" class.
        if shap_values.values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    elif model_type_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer(X_test)
    else:
        raise ValueError(f"No SHAP explainer configured for model type: {model_type_name}")

    return shap_values


def show_global_importance(shap_values, feature_names: list[str]) -> None:
    """Print features ranked by their average impact on predictions.

    I take the mean of the ABSOLUTE SHAP value per feature: this tells
    me which features move the prediction the most on average, whether
    they push it up (toward high-risk) or down (toward not-high-risk).
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(
        ascending=False
    )

    print("--- Global feature importance (mean absolute SHAP value) ---")
    print(importance.to_string())
    print()


def explain_one_example(
    shap_values, X_test: pd.DataFrame, row_index: int, label: str
) -> None:
    """Print a plain, itemized breakdown of one specific prediction.

    Shows each feature's value and its SHAP contribution: positive
    values push the prediction toward "high-risk", negative values push
    it toward "not high-risk".
    """
    print(f"--- Local explanation: {label} (test row {row_index}) ---")

    row_values = X_test.iloc[row_index]
    row_shap = shap_values.values[row_index]
    base_value = shap_values.base_values[row_index]

    contributions = pd.DataFrame(
        {
            "feature_value": row_values,
            "shap_contribution": row_shap,
        }
    ).sort_values("shap_contribution", key=abs, ascending=False)

    print(f"Base value (average model output before features): {base_value:.3f}")
    print(contributions.head(6).to_string())
    print()


def main() -> None:
    df = load_features(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    model = load_trained_model()
    shap_values = compute_shap_values(model, X_train, X_test)

    show_global_importance(shap_values, list(X_test.columns))

    # Pick one real high-risk example and one real not-high-risk example
    # from the test set to walk through in detail.
    y_test_reset = y_test.reset_index(drop=True)
    high_risk_index = y_test_reset[y_test_reset == 1].index[0]
    not_risk_index = y_test_reset[y_test_reset == 0].index[0]

    explain_one_example(shap_values, X_test, high_risk_index, "HIGH-RISK example")
    explain_one_example(shap_values, X_test, not_risk_index, "NOT-high-risk example")


if __name__ == "__main__":
    main()