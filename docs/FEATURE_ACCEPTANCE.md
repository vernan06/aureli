# Aureli feature acceptance

Status reflects automated acceptance evidence collected on 2026-10-26 with Python 3.12.10. “Verified” means an automated local test exercises the behavior; “partially verified” means only part of the journey or a manual path was checked; “blocked” identifies an environmental or product prerequisite.

| Feature | Status | Evidence |
|---|---|---|
| One Streamlit application and five primary sections | Verified | `tests/test_app.py` renders every primary section and each Financial Planner section without exceptions. |
| Authentication, persistence and account isolation | Verified | `tests/test_core.py` verifies password hashing, authentication, owner-scoped CRUD, unauthorized access rejection and guest account cleanup. |
| Transactions, income/expenses, edit and safe deletion controls | Verified | Core tests verify CRUD and user boundaries; AppTest enters a transaction. Deletion now selects a displayed user-owned row and does not ask for an internal ID. |
| Budget utilization, remaining amount and overspend state | Verified | Core test checks spending arithmetic; AppTest confirms the visible remaining amount. |
| Savings goal totals including contributions | Verified | Core contribution test plus AppTest verifies the same ₹15,000 total on Overview and Savings Goals. |
| Loan preview, amortization and repayment recording | Verified | Finance regression tests include ₹100,000 at 12% for 12 months: EMI ₹8,884.88; with ₹2,000 extra, 10 months and ₹5,431.07 interest. AppTest verifies payment defaults ₹8,884.88 total / ₹7,884.88 principal and resulting balance. |
| Placeholder, malformed and duplicate payment rejection | Verified | `tests/test_core.py` rejects ₹0.01/zero-principal placeholders, exact duplicates, malformed amounts, invalid rates and invalid terms. Successful saves rotate the payment form state. |
| Saved-loan repayment scenarios and multiple-loan strategies | Verified | Core tests verify snowball/avalanche calculations and scenario non-mutation. AppTest confirms a saved loan can be selected as a baseline and interest-saved comparison is displayed. |
| Financial twin scenarios do not mutate actual records | Verified | `tests/test_core.py` checks hypothetical changes and confirms stored loans remain unchanged. |
| Cash-flow chart month labels and responsive app styling | Verified | `tests/browser_acceptance.py` exercised the cash-flow chart at desktop and 390px mobile viewport, captured screenshots and found no horizontal overflow. |
| CSV report generation | Partially verified | Reports page renders with owner-scoped records and provides the export control. Download bytes are not directly asserted by the current suite. |
| Authorized EMI models and explanations | Verified | AppTest uses the saved production artifacts to run both real model inference forms; malformed input is rejected. Training metrics and limits are in `ML_RESULTS.md`. |
| Model Laboratory, six-model metrics, plots and MLflow | Verified | AppTest renders the actual six-model comparison and MLflow records from local artifacts/tracking; calibration and residual artifacts are available. |
| Guest demonstration cleanup | Partially verified | Guest user-scoped cleanup is tested at the service boundary; browser journey uses account signup rather than guest sign-out. |
| Legacy schema migration compatibility | Partially verified | Idempotent versioned initialization is implemented; no fixture covers every legacy database version. |
| Live Streamlit startup and browser navigation | Verified | Streamlit reported `http://127.0.0.1:8765`; health endpoint returned HTTP 200. Managed Chromium completed desktop/mobile journeys with no page or console errors. Screenshots are in `.tmp/browser-screenshots/`. |
| Deployment | Not performed | No deployment was attempted, per instruction. |

Run verification from the repository root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m py_compile streamlit_app.py aureli\finance\service.py aureli\loans\service.py aureli\ml\inference.py
.venv\Scripts\python.exe -m pip check
git diff --check
```

Optional browser acceptance uses Python Playwright with its managed Chromium. Install it in the virtual environment, install Chromium to a workspace-local cache, start Streamlit with an isolated `AURELI_DB_PATH`, then pass the actual server-reported URL to `tests/browser_acceptance.py`. See the README for commands.
