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

New accounts start empty. Guest mode is also empty and uses ordinary user-scoped records. The company-provided dataset at `C:\Users\verna\Downloads\emi_prediction_dataset.csv` is authorized by the user for this internship project. The vendor clone and its copies are not used. Training outputs are local and ignored by Git.

```powershell
python -m aureli.ml.train "C:\Users\verna\Downloads\emi_prediction_dataset.csv" --confirm-data-rights --max-rows 120000 --artifact-dir artifacts/models
```

Inspect without training by adding `--inspect-only`. The fit uses a reproducible, proportional stratified sample when valid rows exceed the 120,000-row local memory cap; two-fold cross-validation and modest randomized tuning bound CPU and memory. `--quick` is for pipeline smoke checks only. Training saves artifacts and the data-quality report under `artifacts/models/`, with local MLflow runs under `mlruns/`. Keep generated artifacts and source data private.

## Tests

Run the complete local verification suite with the project environment:

```powershell
python -m unittest discover -s tests -v
```

## Streamlit Community Cloud

For a future deployment, use Python 3.12, `streamlit_app.py`, and both requirements files when MLflow training is needed. Configure durable private storage for user records and model artifacts before accepting real users. Deployment has not been performed.

No public deployment has been performed or verified.
