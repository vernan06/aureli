# Aureli integration status — 2026-10-26

## Product state

Aureli is one Python 3.12 / Streamlit app with five main areas: Overview, EMI Prediction, Model Laboratory, Loan Simulator, and Financial Planner. The planner includes transactions, budgets, savings goals, a financial twin, reports and settings. Shared SQLite persistence uses owner-scoped rows and integer-paise monetary values. Local sign-in uses PBKDF2-HMAC-SHA256; guest records are deleted on guest sign-out.

Finance operations cover categorized income and expenses, recurring obligations, budgets, goal contributions and profile data. One Decimal amortization service powers multiple loans, payment/principal tracking, prepayment and snowball/avalanche comparisons. Digital-twin calculations are pure and do not persist hypothetical values. Cash-flow summaries and recommendations use recorded data. Reports export the available financial records.

The six requested EMI workflows were trained on the user-authorized company CSV: Logistic Regression, Random Forest and XGBoost classification, plus Linear Regression, Random Forest and XGBoost regression. Leakage-safe preprocessing, class weighting, reproducible stratified sampling, train/validation/untouched-test separation, bounded cross-validation/tuning, MLflow tracking, saved pipelines, comparison metrics and UI inference are implemented. See `ML_RESULTS.md` for real test results and limits.

## Data and model limitations

The user confirmed that the company supplied `C:\Users\verna\Downloads\emi_prediction_dataset.csv` and authorized its use for the internship project. The source CSV remains unchanged. All 404,800 rows had valid targets and no exact duplicate rows; 120,000 were sampled proportionally due local compute limits, and the other 284,800 valid rows were not included in this training run. The file fingerprint, cleaning and distributions are in the generated quality report at `artifacts/models/data_quality_report.md`.

The dataset does not include its target-generation specification. The regression champion's held-out same-source R² is 0.9717; this may reflect deterministic label construction and does not establish generalization. High_Risk class precision/recall for the selected classifier is 0.174/0.512. Class outputs are model estimates, not calibrated approval probabilities or lending decisions. Company label documentation and external validation are required before production reliance.

## Source and license disposition

All eight vendor checkouts remain preserved under `vendor/source/`; none of their code or assets were copied into Aureli. Existing notices remain in `THIRD_PARTY_NOTICES.md`.

## Verification

- `.venv` is Python 3.12.10; `pip check` reports no broken requirements.
- `python -m unittest discover -s tests -v`: PASS, 18 tests in 84.019 seconds. Coverage includes signup and UI section rendering; authentication and user isolation; transaction CRUD; budgets; savings; recurring obligations; loan math and repayment; non-mutating payoff/twin scenarios; preprocessing, calibration diagnostics; six-model fixture training, MLflow mechanics, artifact persistence and inference.
- Actual company-data training completed six successful model runs and registered two local champion versions. Exact held-out metrics, MLflow experiment/run IDs and artifact locations are documented in `ML_RESULTS.md`.
- `py_compile` passed on the app and core authentication, finance, loan, scenario and ML modules. `pip check` and `git diff --check` passed.
- The current app's AppTest user journey passed, including all top-level areas with trained local artifacts available.
- A separate loopback Streamlit HTTP health check could not be completed in this managed session: the CLI process remained alive without binding to `127.0.0.1:8765`; process inventory access was denied by Windows. Earlier baseline work had a successful loopback probe, but that does not verify the current revision. No startup defect was observed in AppTest; the live socket gate remains unverified due the environment.
- Upstream MLflow, Streamlit bare-mode, scikit-learn and SHAP deprecation/context warnings appeared during checks.

## Deployment

No deployment was performed. Local usage requires no paid service. A hosted deployment still needs durable private storage and deployment-specific security/retention validation. Model target definition and external validation are also prerequisites before production decisions.
