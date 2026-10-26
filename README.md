# Aureli

**Your personal financial intelligence.** Aureli brings transactions, budgets, savings, debt planning, financial scenarios, reports and educational EMI estimates into one local Streamlit application.

## Launch locally

Requires Python 3.12. From this repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-training.txt
streamlit run streamlit_app.py
```

Aureli opens with account creation, sign-in or a temporary guest demonstration. A local SQLite database defaults to `data/aureli.sqlite3`; set `AURELI_DB_PATH` to use a different private path. Guest records are scoped and deleted when ending the guest session. Do not commit local account databases.

## Product areas

- **Overview:** current-month income, expenses and surplus, recorded loan balance, cash-flow chart, savings progress, budget utilization and contextual guidance.
- **EMI Prediction:** guided input groups, saved-model estimates for `emi_eligibility` and `max_monthly_emi`, model-output probabilities where supported, and feature explanations.
- **Model Laboratory:** all six model comparisons, real held-out metrics, evaluation plots, MLflow runs, preprocessing summary and known limitations.
- **Loan Simulator:** live loan estimates, amortization schedule, principal/interest breakdown, payment history, duplicate/placeholder safeguards, saved-loan repayment scenarios and snowball/avalanche strategies.
- **Financial Planner:** transactions, budgets, savings goals, non-mutating financial twin, CSV reports and profile/recurring settings.

All personal finance queries are scoped to the signed-in account. Stored money uses integer paise; fixed-rate amortization uses `Decimal` and rounds each scheduled payment to paise. Hypothetical scenarios do not change actual records.

## Models and data

The company-provided dataset was authorized for this internship project. Its source CSV is not included in Git. The two required trained inference pipelines, their compact champion metadata, and model comparison table are included under `artifacts/models/` for the public demo; full evaluation plots, data profiles and MLflow runs remain local and are not required for inference. Actual metrics, training method, sample design, MLflow experiment IDs and limits are documented in [docs/ML_RESULTS.md](docs/ML_RESULTS.md).

Reproduce the local training run only when you have the authorized CSV at that path:

```powershell
$env:AURELI_DATASET_PATH = "C:\path\to\authorized\emi_prediction_dataset.csv"
python -m aureli.ml.train "$env:AURELI_DATASET_PATH" --confirm-data-rights --max-rows 120000 --artifact-dir artifacts/models
```

Add `--inspect-only` to audit the file without fitting models. A reproducible, proportional stratified sample is used above the 120,000-row local compute cap. Training writes local artifacts to `artifacts/models/` and MLflow runs to `mlruns/`; these paths are ignored by Git. Model outputs are educational only. High_Risk classification performance is weak and the dataset does not document target-generation rules. Do not use predictions for lending decisions.

## Tests and acceptance

Run the automated suite and local checks:

```powershell
python -m unittest discover -s tests -v
python -m py_compile streamlit_app.py aureli\finance\service.py aureli\loans\service.py aureli\ml\inference.py
python -m pip check
git diff --check
```

See [docs/FEATURE_ACCEPTANCE.md](docs/FEATURE_ACCEPTANCE.md) for coverage and unverified gates, [docs/DEMO_WALKTHROUGH.md](docs/DEMO_WALKTHROUGH.md) for a realistic user journey, and [docs/INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md) for local environment results.

### Optional desktop and mobile browser acceptance

The Playwright acceptance script uses its managed Chromium and a disposable SQLite database. In PowerShell, install the test-only browser tooling and start Streamlit in separate terminals:

```powershell
.venv\Scripts\python.exe -m pip install playwright
$env:PLAYWRIGHT_BROWSERS_PATH = ".tmp\playwright-browsers"
.venv\Scripts\python.exe -m playwright install chromium
$env:AURELI_DB_PATH = ".tmp\browser-acceptance.sqlite3"
.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8765
```

Use the localhost URL printed by Streamlit (the browser acceptance was run at `http://127.0.0.1:8765`) in another terminal:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = ".tmp\playwright-browsers"
.venv\Scripts\python.exe tests\browser_acceptance.py --url http://127.0.0.1:8765
```

The script creates a synthetic test account and saves screenshots in `.tmp\browser-screenshots\`. Keep these local test paths out of Git.

## Deployment

The app is prepared for Streamlit Community Cloud with Python 3.12 and `streamlit_app.py` as its entrypoint. Its required champion model files are in the repository. Configure durable private storage before accepting real user records; the default SQLite file is local to an ephemeral app container. No secrets or paid services are required for the demo.

https://aureli-coorix-emi-predict-ai-15th-oct-ai-ml.streamlit.app/
