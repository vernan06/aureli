# Aureli feature acceptance

Status reflects local evidence collected on 2026-10-26 with Python 3.12.10. “Verified” means the behavior has an automated local test; “partially verified” means only a UI or narrower path was exercised. No product ML models have been trained because data reuse rights are unresolved.

| Feature | Status | Evidence |
|---|---|---|
| Python 3.12 application and one Streamlit entrypoint | Partially verified | `.venv` is Python 3.12.10; `tests/test_app.py` exercises signup and all five sections. A native server socket launch is restricted in this environment. |
| Password authentication and user isolation | Verified | `tests/test_core.py`: password hashing/auth and cross-user record access checks. |
| Income/expense transactions, categories, edits/deletes | Verified | `tests/test_core.py`: owner-scoped transaction CRUD, category association, integer-paise handling. |
| Monthly budgets | Verified | `tests/test_core.py`: budget calculations and owner scope. |
| Savings goals and contributions | Verified | `tests/test_core.py`: contributions, progress and ownership checks. |
| Loan amortization, schedules, payment history and balances | Verified | `tests/test_core.py`: zero-interest/final-payment calculation, payment records, principal balance. |
| Prepayment and snowball/avalanche simulation | Verified | `tests/test_core.py`: payoff effect and strategy calculations without mutation. |
| Recurring obligations and cash-flow summaries | Verified | `tests/test_core.py`: scoped recurring records and descriptive history. |
| Digital twin with income/expense/debt/savings scenarios | Verified | `tests/test_core.py`: scenarios are pure calculations and do not persist changes. |
| CSV reports | Partially verified | Streamlit report page renders during `tests/test_app.py`; export includes transactions, budgets, goals, loans/payments, recurring obligations and totals. CSV download content is not directly asserted. |
| ML preprocessing, six model mechanics, MLflow and inference | Verified as pipeline mechanics only | `tests/test_ml_pipeline.py` trains all six estimators on a generated test fixture, checks local MLflow records, saved artifacts and inference. Fixture scores are not product results. |
| Product EMI predictions and real ML metrics | Blocked | The specified internship dataset is in a source repository without a license or separate dataset terms. No authorized training data or product artifacts are present. |
| Guest demo record cleanup | Partially verified | Guest mode and its user-scoped cleanup are implemented; signup flow is automated, guest sign-out cleanup lacks an independent test. |
| Schema migration and preservation of older local records | Partially verified | Idempotent versioned schema initialization is implemented; explicit legacy-schema migration fixture is not part of the acceptance suite. |
| Deployment readiness | Partially verified | Single entrypoint and dependency manifests are prepared. No deployment or durable hosted storage has been configured or attempted. |

Run evidence: the complete unittest command passed all 16 tests. A separate live Streamlit smoke run bound to 127.0.0.1:8765 and returned HTTP 200 from /_stcore/health.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m py_compile streamlit_app.py aureli\auth\service.py aureli\finance\service.py aureli\loans\service.py aureli\intelligence\scenario.py aureli\ml\pipeline.py aureli\ml\inference.py
```
