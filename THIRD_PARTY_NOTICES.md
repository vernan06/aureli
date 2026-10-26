# Third-party source audit and notices

The directories under `vendor/source/` were inspected in place. They remain excluded from the Aureli product build and Git history. No source files, databases, serialized model artifacts, UI assets, or data files were copied from those repositories into Aureli. Aureli's current core services and UI are original implementations. Existing vendor directories and nested Git repositories are preserved as supplied.

| Repository | License evidence | Assessment | Aureli reuse |
|---|---|---|---|
| `finance` / MoneyFlow | MIT, `vendor/source/finance/LICENSE` (Leo Teme Teme, 2026) | Compatible for reuse with notice preservation; broad, newer dependencies and existing package structure were not needed for this initially empty Aureli codebase. | None copied. |
| `experience` / Expensy | No root license found | Reference-only; do not copy code or assets without permission. | None. |
| `risk` / Loan Approval Prediction | MIT, `vendor/source/risk/LICENSE` (Ana Nicuesa, 2026) | Compatible for reuse with notice preservation, but its binary approval target is not the required EMI eligibility target. | None copied; models and data not used. |
| `experiments` / Loan Default ML Pipeline | GNU GPL v3, `vendor/source/experiments/LICENSE` | Copyleft obligations and unrelated German-credit default target make direct reuse unsuitable for this deployment. | None. |
| `emi_ml` / EMI Prediction System | No root license found; no separate dataset terms located | Reference-only. Existing model binaries and EMI data are not used; no predictions or model metrics are claimed. | None. |
| `ml_training` / MLOps Platform | MIT, `vendor/source/ml_training/LICENSE` (Hemin Vora, 2026) | Compatible for reuse with notice preservation; its FastAPI, Postgres, Docker, and external monitoring infrastructure conflict with Aureli's lightweight single-app target. | None copied. |
| `loan_core` / PyLoan | MIT, `vendor/source/loan_core/LICENSE` (Da.Ki.X, 2021) | Compatible for reuse with notice preservation. Its general date/day-count API was not needed for Aureli's fixed-rate monthly product. | None copied; Aureli loan calculations are independently implemented. |
| `loan_ui` / Loan Calculator | No root license found | Reference-only; do not copy code or assets without permission. | None. |

## Runtime dependencies

The app depends on Streamlit and Plotly as declared in `requirements.txt`; consult their upstream distribution notices for those packages. No vendor application source is bundled. `requirements-training.txt` is optional and only for a separate, local training environment.
