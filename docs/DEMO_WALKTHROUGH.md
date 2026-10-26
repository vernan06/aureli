# Aureli local demo walkthrough

Start the local app with `.venv\Scripts\streamlit.exe run streamlit_app.py`, then choose **Continue in guest demo mode**. This walkthrough uses a new empty demo workspace. For a persistent local journey, create an account instead.

1. **Overview:** see current-month income, expenses, surplus and recorded loan principal. Record data to populate the formatted monthly cash-flow chart. Inspect contribution-inclusive savings progress, budget remaining/utilization, personal guidance and model readiness.
2. **Financial Planner → Transactions:** add salary income and categorized expenses. Edit an entry, then remove a selected row using the safe selection control. Return to Overview and review the updated totals.
3. **Budgets:** create a groceries or rent limit. Aureli displays amount spent, utilization percentage, remaining amount, progress and an overspending message when applicable.
4. **Savings Goals:** create an emergency fund target and contribution. Confirm the total matches Overview. Review history, enter a planned monthly contribution to estimate time to target, and update the goal if needed.
5. **Settings:** save typical monthly income and emergency-fund amount, then add and pause/reactivate recurring obligations.
6. **Loan Simulator:** enter a name, principal, rate, term, start date and optional extra principal. See the EMI, total repayment, interest and duration preview update while editing. Save the loan to inspect the remaining-balance schedule, payment breakdown, progress and history.
7. **Record a loan payment:** use the lender statement to confirm the principal component. Aureli pre-fills an estimate, previews the outstanding balance, blocks zero-principal/placeholder and duplicate submissions, then refreshes the next payment form after recording.
8. **Compare payoff:** select an existing loan, set an extra monthly principal amount and compare estimated interest/time saved. With multiple loans, compare snowball and avalanche strategies. These simulations do not alter saved balances.
9. **Financial Twin:** change income, expenses, borrowing, extra repayment and savings inputs. Review projected surplus, EMI, debt service and savings. Check that actual records remain unchanged.
10. **EMI Prediction:** enter known details in the guided financial sections and run both saved models. Inspect estimated class, model scores, maximum EMI and feature explanations. Results are educational estimates, not lending decisions.
11. **Model Laboratory:** show the six actual model metrics, champion comparisons, confusion/calibration and regression evaluation plots, MLflow records, dataset summary, preprocessing method and limitations.
12. **Reports and sign-out:** download a CSV report, then end the guest session. Guest records are removed; another account’s data remains isolated.

This is a written guide to the implemented local product. No demo video was recorded. The managed environment did not permit a current live HTTP/browser check, so human acceptance should include launching the app in a normal local Windows session and reviewing desktop/mobile sizing.
