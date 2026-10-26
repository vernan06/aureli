# Aureli agent handoff

## Current state

Aureli is a single local Streamlit application using Python 3.12, SQLite, one finance service, one loan calculation service, and one ML pipeline. Product features and verification evidence are tracked in FEATURE_ACCEPTANCE.md. A realistic local user journey is in DEMO_WALKTHROUGH.md.

The full dependency environment is in .venv. Launch from C:\aureli with:

    .venv\Scripts\streamlit.exe run streamlit_app.py

Install dependencies with:

    .venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-training.txt

Run local verification with:

    .venv\Scripts\python.exe -m unittest discover -s tests -v

## ML/data boundary

The six-model workflow and MLflow integration are implemented and mechanically verified on a generated test fixture only. Do not report fixture metrics as product metrics. The specified internship dataset's reuse rights are unresolved: its vendor repository has no root license, and no separate dataset terms were found. No product training, output or saved model artifacts exist. Obtain the authorized dataset and documentary permission before invoking the CLI with --confirm-data-rights.

## Preservation and repository

Vendor repositories under vendor/source/ and their nested Git metadata have been preserved and excluded from application code. No vendor source has been copied. The pre-existing dirty .gitignore edit was preserved and not included in the commit. No local financial database, models, MLflow runs, or generated test artifacts should be committed. No deployment was attempted.

## Remaining external prerequisites

- Authorized internship dataset and explicit permission for model training.
- Before any real deployment, choose durable private storage and validate deployment-specific security and retention behavior.
- No demo video was recorded; DEMO_WALKTHROUGH.md is the actual-feature walkthrough.

Do not claim model predictions, product metrics, or public deployment as complete until the respective prerequisites are satisfied.
