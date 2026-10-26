# Aureli integration status — 2026-10-26

## Product state

The repository has one Python 3.12 / Streamlit app with five top-level sections: Overview, EMI Prediction, Model Laboratory, Loan Simulator, and Financial Planner. The planner contains transactions, budgets, savings goals, a financial twin, reports, and settings. Shared SQLite persistence uses owner-scoped rows and integer-paise monetary values. Local sign-in uses PBKDF2-HMAC-SHA256; temporary guest accounts are deleted on sign-out.

Finance operations cover categorized income and expenses, recurring obligations, budget spending, goal contributions, profile data, and CSV export. One Decimal amortization service powers multiple loans, payment/principal tracking, prepayment and snowball/avalanche comparisons. Digital-twin calculations are pure and do not persist simulated values. Recommendations and cash-flow summaries use recorded data.

The EMI system defines Logistic Regression, Random Forest, and XGBoost classifiers plus Linear Regression, Random Forest, and XGBoost regressors; train-only preprocessing; train/validation/test separation; imbalance-aware classification; validation-based champion selection; held-out metrics; MLflow experiments and model registry; saved Joblib pipelines; feature importance and SHAP artifacts where supported; and local inference/UI integration. Pipeline mechanics have been executed using a generated test fixture. It is not product data and its metrics are not product results.

## External data blocker

No authorized copy of the specified internship dataset or documented reuse permission is available in this checkout. The vendor EMI source repository has no root license, and no separate dataset terms were found during the source audit. Its data and model binaries have not been used. No Aureli product models, predictions, or real product metrics exist. Training requires an explicit rights affirmation and refuses without it. Actual EMI model availability is blocked until the data and its rights are provided.

## Source and license disposition

All eight vendor checkouts were reviewed and remain untouched under vendor/source/; none of their code or assets were copied. The compatible MIT repositories (MoneyFlow, Loan Approval Prediction, MLOps Platform, PyLoan) were unnecessary to integrate with Aureli services. Expensy, EMI Prediction System, and Loan Calculator have no root license and remain reference-only. Loan Default ML Pipeline is GPL-3.0 and targets a different outcome. See THIRD_PARTY_NOTICES.md.

## Verification

- .venv is Python 3.12.10; pinned application and ML dependencies installed successfully.
- tests/test_core.py covers authentication, owner isolation, transaction CRUD, budget calculations, savings contributions, recurring obligations, loan schedules/payments, payoff strategies, cash-flow summaries, recommendations, and non-mutating scenarios.
- tests/test_app.py uses Streamlit AppTest to register, add a transaction, and render all top-level sections and planner pages.
- tests/test_ml_pipeline.py validates target schema, exercises all six estimators on a generated test-only fixture, checks MLflow records, registered models, saved pipelines, and inference.
- Python compilation and git diff --check passed.
- A real loopback Streamlit server started on port 8765; /_stcore/health returned HTTP 200 ok; server stopped after the check.
- Streamlit bare-mode, MLflow, scikit-learn, and SHAP emitted upstream/context warnings during verification.
- `.venv\\Scripts\\python.exe -m unittest discover -s tests -v`: PASS, all 16 tests in 113.3 seconds, including the UI journey, finance/security checks, and six-model/MLflow/inference fixture workflow.

## Deployment

No deployment was performed. The app and dependency manifests are prepared for local use; durable private hosted storage, data-rights clearance, and actual product model training remain prerequisites before public use. No paid service is required by the app code.
