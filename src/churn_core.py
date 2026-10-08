"""Shared building blocks for the Telco customer churn project.

Everything that must be identical for all three team members (random seed,
split ratio, column groups, preprocessing recipe) is defined once, here.
Import from this module instead of re-typing these values in a notebook.
"""
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INTERNET_DEPENDENT_FEATURES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])


def load_raw_customers(path=RAW_DATA_PATH):
    """Read the original CSV without changing anything."""
    return pd.read_csv(path)


def clean_customers(customers_raw):
    """Return a cleaned copy of the raw table, indexed by customerID.

    Steps: strip stray whitespace, convert TotalCharges to float, fill the
    blank TotalCharges of brand-new customers (tenure == 0) with 0, and merge the
    labels "No internet service" / "No phone service" into "No".
    Those labels only repeat InternetService == "No" and PhoneService == "No";
    keeping them would create identical dummy columns and unstable coefficients.
    All rules are fixed by business logic, not estimated from data, so applying
    them before the train/test split cannot leak test information.
    """
    if customers_raw[ID_COLUMN].duplicated().any():
        raise ValueError("customerID must be unique.")

    customers = customers_raw.copy()
    for column in customers.select_dtypes(exclude="number").columns:
        customers[column] = customers[column].str.strip()

    total_charges_is_blank = customers["TotalCharges"].eq("")
    customer_never_billed = customers["tenure"].eq(0)
    if not total_charges_is_blank.equals(customer_never_billed):
        raise ValueError("Blank TotalCharges no longer matches tenure == 0.")

    total_charges = pd.to_numeric(
        customers["TotalCharges"].where(~total_charges_is_blank), errors="raise"
    )
    customers["TotalCharges"] = total_charges.fillna(0.0).astype("float64")

    _merge_redundant_service_labels(customers)
    return customers.set_index(ID_COLUMN)


def _merge_redundant_service_labels(customers):
    """Replace "No internet service" / "No phone service" by "No", in place."""
    redundant_label_rules = [
        (INTERNET_DEPENDENT_FEATURES, "No internet service", customers["InternetService"].eq("No")),
        (["MultipleLines"], "No phone service", customers["PhoneService"].eq("No")),
    ]
    for columns, redundant_label, implied_by_other_column in redundant_label_rules:
        for column in columns:
            if not customers[column].eq(redundant_label).equals(implied_by_other_column):
                raise ValueError(f"'{redundant_label}' in {column} is not fully implied by another column.")
            customers[column] = customers[column].replace(redundant_label, "No")


def split_train_test(customers):
    """Stratified 80/20 split shared by the whole team.

    Returns X_train, X_test, y_train, y_test where y is 1 for churn, 0 otherwise.
    """
    features = customers[ALL_FEATURES]
    target = customers[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype("int64")
    return train_test_split(
        features, target,
        test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE,
    )


def build_preprocessor():
    """Unfitted ColumnTransformer: scale numeric, one-hot encode categorical.

    Always put it inside a Pipeline (or imblearn Pipeline) so that fit() only
    ever sees training data, including inside every cross-validation fold.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def build_model_pipeline(estimator):
    """Preprocessing + estimator as one object, ready for fit/predict/GridSearchCV."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def get_feature_names(fitted_pipeline):
    """Column names produced by the fitted preprocessing step."""
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())


def compute_basic_metrics(y_true, y_pred):
    """Accuracy and churn-class (label 1) precision, recall, F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def make_logistic_regression(penalty, C=1.0):
    """L1 or L2 logistic regression that runs without warnings on scikit-learn 1.2+.

    scikit-learn 1.8 replaced `penalty` with `l1_ratio` (1.0 means L1, 0.0 means L2).
    liblinear is used because it supports both penalties on this small dataset.
    """
    if penalty not in ("l1", "l2"):
        raise ValueError("penalty must be 'l1' or 'l2'.")
    if SKLEARN_VERSION >= (1, 8):
        penalty_settings = {"l1_ratio": 1.0 if penalty == "l1" else 0.0}
    else:
        penalty_settings = {"penalty": penalty}
    return LogisticRegression(
        C=C, solver="liblinear", max_iter=1000, random_state=RANDOM_STATE, **penalty_settings
    )
