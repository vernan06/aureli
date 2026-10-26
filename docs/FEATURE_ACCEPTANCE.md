# Aureli feature acceptance

Status reflects local evidence collected on 2026-10-26 with Python 3.12.10. “Verified” means the behavior has automated local test evidence; “partially verified” means only a narrower automated or manual path was exercised; “blocked” identifies an external prerequisite.

| Feature | Status | Evidence |
|---|---|---|
| Python 3.12 application and one Streamlit entrypoint | Verified | `.venv` is Python 3.12.10; AppTest registers, records a transaction and renders all five sections; current real server health check is recorded in INTEGRATION_STATUS.md. |
| Password authentication and user isolation | Verified | `tests/test_core.py`: password hashing/authentication and cross-user record-access checks. |
| Income/expense transactions, categories, edits/deletes | Verified | `tests/test_core.py`: owner-scoped transaction CRUD and integer-paise handling; `tests/test_app.py` adds a transaction through the UI. |
| Monthly budgets | Verified | `tests/test_core.py`: calculations and owner scope. |
| Savings goals and contributions | Verified | `tests/test_core.py`: contributions, progress and ownership. |
| Loan amortization, schedules, payment history and balances | Verified | `tests/test_core.py`: schedule, payment records and principal balance. |
| Prepayment and snowball/avalanche strategies | Verified | `tests/test_core.py`: payoff changes and strategy calculations without mutation. |
| Recurring obligations and cash-flow summaries | Verified | `tests/test_core.py`: scoped recurring records and summaries. |
| Financial twin scenarios | Verified | `tests/test_core.py`: scenario calculations do not persist or mutate saved financial records. |
| CSV report generation | Partially verified | Report view renders in `tests/test_app.py`; export implementation includes transactions, budgets, goals, loans/payments, recurring obligations and totals. Download bytes are not independently asserted. |
| Data cleaning and preprocessing | Verified | `tests/test_ml_pipeline.py`: schema, mixed types, missing-value handling and normalization; actual-data audit in `artifacts/models/data_quality_report.md`. |
| Six model training workflows and MLflow | Verified | Actual company dataset generated six FINISHED MLflow runs and two registered champion versions. Exact IDs/metrics are in `docs/ML_RESULTS.md`; synthetic-fixture mechanics remain covered by `tests/test_ml_pipeline.py`. |
| Saved preprocessing pipelines and app inference | Verified | Two joblib pipelines and metadata exist under `artifacts/models/`; inference was exercised on a held-out manual input and UI pages render with artifacts. No user prediction should be interpreted as a lending decision. |
| Classification quality for minority High_Risk class | Partially verified | Held-out metrics are real and reported; selected model precision 0.174 and recall 0.512 are weak. No claim of production suitability. |
| Regression target generalization | Blocked | Held-out Random Forest R² is 0.9717, but target-generation specification is absent. Company label documentation and external validation are required for production reliance. |
| Guest demo cleanup | Partially verified | Guest mode is implemented and UI paths render; independent sign-out cleanup assertion is not in the current suite. |
| Legacy schema/data migration | Partially verified | Versioned idempotent initialization is implemented; no dedicated legacy-schema migration fixture covers every earlier database state. |
| Deployment readiness | Partially verified | Single entrypoint and dependency manifests are prepared. No deployment occurred; durable hosted private storage and deployment security/retention configuration remain to be selected. |

Verification command:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
