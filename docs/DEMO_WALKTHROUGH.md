# Aureli local demo walkthrough

Launch the local app with `.venv\Scripts\streamlit.exe run streamlit_app.py`, then select **Continue in guest demo mode**. The guest workspace begins empty and is deleted when the guest session ends. The signed-in product uses local SQLite records, and the trained EMI pipelines are loaded from `artifacts/models/`.

1. **Overview:** review current-month income, expenses and surplus, recorded loan principal, cash-flow trend, savings progress, budgets and guidance. Empty sections explain what data is needed.
2. **Financial Planner → Transactions:** add income and categorized expenses, then edit or remove a ledger entry. Return to Overview and see recorded totals update.
3. **Budgets:** create a monthly groceries or rent limit and compare the saved transaction spend with the remaining amount.
4. **Savings Goals:** add a goal and dated contribution; check contribution history and progress.
5. **Settings:** save typical monthly income and emergency funds. Add a recurring obligation such as rent, then pause or reactivate it.
6. **Loan Simulator:** create a loan, inspect its EMI and amortization schedule, record a lender-confirmed principal payment, and inspect the remaining principal. Compare snowball and avalanche payoff orders; simulations do not alter the saved loans.
7. **Financial Twin:** change hypothetical income, expenses, savings, new borrowing or prepayments. Review projected surplus, EMI, debt-to-income and savings. These scenario inputs never mutate actual records.
8. **Reports:** download a CSV containing available saved transactions, totals, budgets, goals, loans, payment history and recurring obligations.
9. **EMI Prediction:** enter a financial profile and view the trained model's eligibility class and estimated maximum monthly EMI, with explanatory factors where available. Results are model estimates, not lending decisions. The regression range and target-label limitations are described in `docs/ML_RESULTS.md`.
10. **Model Laboratory:** inspect the six-model comparison, held-out test metrics, confusion matrices, calibration diagnostics and regression residual/actual-vs-predicted views. Metrics are from the authorized company CSV and are not synthetic fixture scores.
11. **End guest session:** sign out; the guest's stored records are removed. A regular local account is available for persistent local use.

This is a written walkthrough of implemented functionality. No demo video was recorded.
