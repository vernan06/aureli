# Aureli EMI model results

**Training run:** 2026-10-26, Python 3.12.10, scikit-learn 1.5.2, XGBoost 2.1.4, MLflow 2.14.3. The user confirmed that the supplied company dataset is authorized for this internship project. The original CSV was read only and remains unchanged.

## Dataset and evaluation design

- File: `emi_prediction_dataset.csv`; SHA-256: `46a52dcbc80b8d736c5a42e9fa5173289e66bb9d30dd03e4e2f33e2b9c1a53c4`.
- Original rows: 404,800; rows retained after cleanup: 404,800; exact duplicates: 0; invalid/missing target exclusions: 0.
- 120,000 rows were selected by reproducible proportional stratification over `emi_eligibility` and `max_monthly_emi` quantile bands due local memory/CPU limits. The other 284,800 valid rows were not sampled into this run; they were not classified as bad data.
- For each target: training 72,000 rows, validation 24,000 rows, untouched test 24,000 rows. The selected pipeline is refit on training plus validation; test data is not used to tune or select it. Two-fold CV and two randomized candidates per model were used.
- Classification class counts in the original dataset: Not_Eligible 312,868; Eligible 74,444; High_Risk 17,488. Largest-to-smallest imbalance ratio: 17.89:1. Logistic Regression and Random Forest use balanced class weights; XGBoost uses balanced sample weights.
- Missing values remain missing and are imputed within training folds. There were no duplicates or target rows to exclude. Numeric string serialization errors and gender variants were normalized as detailed in [data_quality_report.md](../artifacts/models/data_quality_report.md).

## Classification held-out test results

| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC AUC (OvR) | Multiclass Brier | Log loss | ECE (10 bins) | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7797 | 0.6259 | 0.8323 | 0.9203 | 0.2877 | 0.5391 | 0.0580 | 27.79 |
| Random Forest | 0.8427 | **0.6621** | 0.8707 | **0.9373** | **0.2397** | **0.4473** | 0.1186 | 72.87 |
| XGBoost | 0.7679 | 0.6239 | 0.8250 | 0.9225 | 0.3232 | 0.5829 | 0.1473 | 52.88 |

Precision and recall for each class:

| Model | Eligible P / R | High_Risk P / R | Not_Eligible P / R |
|---|---:|---:|---:|
| Logistic Regression | 0.814 / 0.729 | 0.137 / 0.660 | 0.983 / 0.799 |
| Random Forest (selected) | 0.800 / 0.814 | 0.174 / 0.512 | 0.979 / 0.868 |
| XGBoost | 0.806 / 0.744 | 0.136 / 0.699 | 0.989 / 0.777 |

The selected classifier maximizes validation macro-F1. High_Risk precision remains low for all three models, and Random Forest High_Risk recall is 0.512. XGBoost has higher High_Risk recall (0.699) but lower macro-F1 and only 0.136 High_Risk precision. The confusion matrices and calibration plots are saved per model under `artifacts/models/evaluation/emi_eligibility/`. Probabilities are model outputs, are not calibrated estimates of approval odds, and must not be used as lending decisions.

## Regression held-out test results

| Model | MAE (INR) | RMSE (INR) | R² | Residual mean (INR) | Max absolute residual (INR) | Train seconds |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 2,977.01 | 4,145.03 | 0.7172 | 43.85 | 48,832.50 | 18.90 |
| Random Forest (selected) | **613.86** | **1,312.34** | **0.9717** | 14.30 | 30,451.43 | 305.41 |
| XGBoost | 1,413.15 | 2,441.74 | 0.9019 | 15.42 | 50,517.80 | 24.73 |

The champion minimizes validation RMSE. Its prediction is bounded to the target range observed in the training sample (₹500–₹88,200). Actual-versus-predicted and residual plots are under `artifacts/models/evaluation/max_monthly_emi/`.

## Target leakage and interpretation limits

The two target columns are removed from every feature matrix. The audit found no exact feature copy of either target. The strongest single Pearson correlation with `max_monthly_emi` was 0.485 (`groceries_utilities`); no single feature duplicated the target. However, the label-generation rules were not included with the CSV. The very high regression R² may reflect a deterministic affordability formula built from the same financial inputs. A random same-source holdout cannot establish performance on future customers or a lending population. Treat predictions as estimates within this dataset’s label definition; obtain the company’s target-generation documentation and external validation before relying on them beyond local demonstration.

No financial values were fabricated or capped. The test set has not been used for calibration, feature selection, or champion selection. The logged ECE/Brier/log-loss and reliability diagrams are diagnostic; no probability calibration transform was fitted.

## Saved outputs

- Inference pipelines (including preprocessing): `artifacts/models/emi_eligibility.joblib`, `artifacts/models/max_monthly_emi.joblib`.
- Champion metadata and full test metrics: corresponding `*.metadata.json` files in `artifacts/models/`.
- Six-model comparison: `artifacts/models/model_comparison.csv`.
- Full data quality profile/report: `artifacts/models/dataset_profile.json`, `artifacts/models/data_quality_report.md`.
- Local MLflow experiments: `mlruns/`; classification experiment ID `493838325564776429`, regression experiment ID `317780232923495581`. Each of the six successful estimator runs records parameters, test metrics, data fingerprint, preprocessing/model artifacts, evaluation plots, SHAP artifacts where supported, and training duration. Champion model versions are registered locally as `aureli-emi_eligibility` v1 and `aureli-max_monthly_emi` v1.
- Model outputs and MLflow files are local, ignored by Git, and not committed.

Reproduce from the repository root:

```powershell
.venv\Scripts\python.exe -m aureli.ml.train "C:\Users\verna\Downloads\emi_prediction_dataset.csv" --confirm-data-rights --max-rows 120000 --artifact-dir artifacts/models
```
