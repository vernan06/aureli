# Aureli

**Your personal financial intelligence.** One local Streamlit application for EMI modeling, personal cash flow, budgets, savings goals, loans, simulations, and reports.

## Run locally

Requires Python 3.12. From this repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-training.txt
streamlit run streamlit_app.py
```

The app offers account creation and a temporary guest demo. SQLite defaults to `data/aureli.sqlite3`; set `AURELI_DB_PATH` to another private path when needed. Guest demo records are deleted when the guest session ends. Never commit the local database.

## Architecture

- `streamlit_app.py` is the only UI entrypoint. It presents Overview, EMI Prediction, Model Laboratory, Loan Simulator, and Financial Planner sections.
- `aureli/auth` provides one PBKDF2 password authentication system.
- `aureli/persistence` owns SQLite schema creation and connection settings.
- `aureli/finance` owns integer-paise transactions, categories, budgets, recurring obligations, savings goals, and profiles.
- `aureli/loans` owns Decimal amortization, payment history, principal balances, and snowball/avalanche comparisons.
- `aureli/intelligence` owns stored-fact recommendations, cash-flow summaries, and non-mutating plan scenarios.
- `aureli/ml` owns leakage-safe preprocessing, the six estimator workflows, local MLflow tracking, persisted artifacts, and inference.
- Stored money uses integer paise; loan math uses Decimal and rounds each scheduled payment to paise.
- Records are scoped to the signed-in user in every data query.

New accounts start empty. Guest mode is also empty and uses ordinary user-scoped records. EMI prediction remains unavailable until the specified internship dataset and its reuse rights are verified. Vendor datasets are excluded from training. Do not use `--confirm-data-rights` unless you have verified the data license/permission. Training command (after rights are cleared):

```powershell
python -m aureli.ml.train --data path\to\authorized-internship-data.csv --confirm-data-rights
```

Use `--quick` for a pipeline smoke run only; do not report its synthetic fixture metrics as product results. Training writes artifacts under `models/` and MLflow local runs under `mlruns/` (or the configured tracking URI). Keep generated artifacts and data private.

## Tests

Run the complete local verification suite with the project environment:

```powershell
python -m unittest discover -s tests -v
```

## Streamlit Community Cloud

For a future deployment, use Python 3.12, `streamlit_app.py`, and both requirements files when MLflow training is needed. Configure durable private storage for user records and model artifacts before accepting real users. Deployment has not been performed.

No public deployment has been performed or verified.
