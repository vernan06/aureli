# Aureli agent handoff

## Current state

Aureli is one local Streamlit application on Python 3.12 with SQLite, owner-scoped finance services, Decimal loan calculations, non-mutating financial scenarios, report export, and a six-model EMI pipeline. Product state and evidence are in [FEATURE_ACCEPTANCE.md](FEATURE_ACCEPTANCE.md); model results and limitations are in [ML_RESULTS.md](ML_RESULTS.md); a realistic UI journey is in [DEMO_WALKTHROUGH.md](DEMO_WALKTHROUGH.md).

The prepared `.venv` uses Python 3.12.10. From `C:\aureli`:

    .venv\Scripts\streamlit.exe run streamlit_app.py

To install dependencies:

    .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-training.txt

Run verification:

    .venv\Scripts\python.exe -m unittest discover -s tests -v

## Models and data

The user authorized the company-supplied `C:\Users\verna\Downloads\emi_prediction_dataset.csv` for the internship Aureli project. The source file was read but not modified. All six requested workflows completed: Logistic Regression, Random Forest, XGBoost classification and Linear Regression, Random Forest, XGBoost regression. Each has a saved preprocessing/model pipeline and a FINISHED MLflow run; local champion versions are registered. Artifacts are under ignored `artifacts/models/`, and MLflow tracking is under ignored `mlruns/`. No trained model binaries or source dataset are committed.

To reproduce the capped local training run:

    .venv\Scripts\python.exe -m aureli.ml.train "C:\Users\verna\Downloads\emi_prediction_dataset.csv" --confirm-data-rights --max-rows 120000 --artifact-dir artifacts/models

Exact test metrics, dataset fingerprint, sampling design and limitations are in `docs/ML_RESULTS.md`. High_Risk classification performance is weak; the regression target generation rule is undocumented. Model outputs are advisory and require target documentation and external validation before production reliance. Earlier aborted diagnostic runs remain marked FAILED in MLflow; they are distinct from the six successful runs documented in ML_RESULTS.md.

## Preservation and deployment

Vendor repositories and nested Git metadata are preserved; no vendor source was copied into Aureli. The pre-existing `.gitignore` edit was kept outside Aureli commits. Never commit local databases, source data, model artifacts or MLflow runs. No deployment was performed. Hosted deployment still needs a durable private database/artifact storage design and deployment-specific security and retention validation.

`DEMO_WALKTHROUGH.md` is a written walkthrough of actual local functionality. No demo video was recorded.
