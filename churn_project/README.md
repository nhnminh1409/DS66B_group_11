# Telco Customer Churn Prediction (Statistical Learning)

## Structure
| Path | Content |
|---|---|
| `data/raw/telco_customer_churn.csv` | Original Kaggle/IBM Telco Customer Churn file, untouched |
| `data/processed/train.csv`, `test.csv` | Shared stratified 80/20 split (index = customerID, `Churn` 1/0) |
| `src/churn_core.py` | Shared constants, cleaning, split, preprocessing pipeline (single source of truth) |
| `notebooks/01_tan_cleaning_pipeline_baseline.ipynb` | Tan: Tasks 1.1-1.4, 2.1, 2.2 |
| `artifacts/` | Saved models, baseline and tuning tables, coefficient table |
| `reports/figures/` | Figures for the report |

## Setup
```
pip install -r requirements.txt
jupyter notebook   # open notebooks/01_..., Kernel -> Restart & Run All
```

## Rules for all members
1. Never copy the split or preprocessing code. Import it: `import churn_core as core`.
2. `core.RANDOM_STATE = 42`, `core.TEST_SIZE = 0.20`, `core.N_CV_FOLDS = 5`.
3. Preprocessing and SMOTE go inside a Pipeline so they are fitted inside each CV fold only.

## Quick start for a teammate notebook
```python
import sys; from pathlib import Path
sys.path.append(str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/"src"/"churn_core.py").exists())/"src"))
import churn_core as core

customers = core.clean_customers(core.load_raw_customers())
X_train, X_test, y_train, y_test = core.split_train_test(customers)

# Yen: imblearn.pipeline.Pipeline([("preprocess", core.build_preprocessor()), ("smote", SMOTE(random_state=core.RANDOM_STATE)), ("model", ...)])
# NMinh: model = joblib.load(core.ARTIFACTS_DIR / "best_logreg_pipeline.joblib"); names = core.get_feature_names(model)
```
