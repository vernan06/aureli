# Aureli integration status — 2026-10-26

## Product state

Aureli is one Python 3.12 / Streamlit application with Overview, EMI Prediction, Model Laboratory, Loan Simulator and Financial Planner areas. Shared SQLite persistence uses owner-scoped rows and integer-paise money. Local sign-in uses PBKDF2-HMAC-SHA256; guest records are removed on guest sign-out.

The finance modules cover categorized income and expenses, recurring obligations, budgets, goal contributions, multiple loans, Decimal amortization, repayment tracking, prepayments, snowball and avalanche comparisons, cash-flow summaries, recommendations, non-mutating financial twin scenarios and CSV reports. A single UI connects these features to the shared services.

All six requested EMI workflows were trained on the user-authorized company CSV: Logistic Regression, Random Forest and XGBoost classification, plus Linear Regression, Random Forest and XGBoost regression. Leakage-safe preprocessing, class weighting, reproducible sampling, train/validation/test separation, MLflow tracking, saved pipelines, comparison metrics and UI inference are implemented. Exact held-out metrics and artifact/run locations are documented in `ML_RESULTS.md`.

## Data and model limitations

The user confirmed and authorized the company-supplied CSV for this internship project. The source CSV remains unchanged. All 404,800 rows had valid targets and no exact duplicate rows; the training run used a proportional sample of 120,000 rows because of local compute limits. The file fingerprint, cleaning and distributions are recorded in the local data-quality report.

The dataset does not include its target-generation specification. The regression champion's held-out same-source R² is 0.9717; this does not establish generalization. High_Risk class precision/recall for the selected classifier is 0.174/0.512. These are educational model estimates, not calibrated approval probabilities or lending decisions. Company label documentation and external validation are required before production reliance.

## Source and license disposition

All eight vendor checkouts remain preserved under `vendor/source/`; none of their code or assets were copied into Aureli. Existing attribution notices remain in `THIRD_PARTY_NOTICES.md`.

## Verification

- `.venv` uses Python 3.12.10. `pip check`, `py_compile` and `git diff --check` were run after the latest fixes.
- `.venv\Scripts\python.exe -m unittest discover -s tests -v`: PASS, 23 tests in 73.619 seconds. Coverage includes authentication and isolation, finance CRUD, savings/budgets, loan calculations and payments, twin scenarios, preprocessing, fixture training, MLflow mechanics, artifact persistence, inference and Streamlit page rendering.
- A real Streamlit server started at the server-reported `http://127.0.0.1:8765`; its loopback health endpoint returned HTTP 200.
- `tests/browser_acceptance.py` ran with Playwright's managed Chromium against that localhost URL. Desktop and 390px mobile journeys covered account creation, transaction entry, budget/savings synchronization, loan estimates and repayment, comparison scenarios, both model inferences and invalid-number feedback, Model Laboratory metrics/plots, Overview charts, and mobile planner navigation. No browser page or console errors were recorded, and the mobile page had no horizontal overflow. Screenshots are local under `.tmp/browser-screenshots/` and are intentionally not committed.
- Browser automation used the installed Python Playwright package and managed Chromium. No browser-control bypass was used.
- Test output included upstream deprecation warnings from MLflow, scikit-learn/SciPy and SHAP, plus expected Streamlit bare-mode context warnings; no test failures resulted.
- The app's machine-learning test uses the trained local artifacts; training metrics are actual results documented in `ML_RESULTS.md`.

## Deployment

Deployment files target Streamlit Community Cloud using Python 3.12 and `streamlit_app.py`. Champion inference packages, their metadata and comparison table are included; source data, MLflow records and optional plot files are excluded. No deployment has been created. Community Cloud's SQLite filesystem is ephemeral, so hosted use with real financial records still needs durable private storage and deployment-specific security and retention validation. Model target definition and external validation are prerequisites before production decisions.
