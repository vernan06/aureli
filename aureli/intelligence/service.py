"""Deterministic personal-finance guidance derived from the signed-in user's records."""
from datetime import date
from decimal import Decimal
from aureli.finance.service import budgets, monthly_summary, profile
from aureli.loans.service import loan_summary, user_loans

def recommendations(user_id: int, month: str | None = None):
    snapshot=monthly_summary(user_id,month)
    p=profile(user_id)
    income=snapshot["income"] or (p["monthly_income_minor"] if p else 0)
    expenses=snapshot["expenses"]
    loans=user_loans(user_id)
    loan_emi=sum(int(loan_summary(Decimal(row["principal_minor"])/100,row["annual_rate"],row["term_months"],Decimal(row["extra_payment_minor"])/100)["emi"]*100) for row in loans)
    result=[]
    surplus=income-expenses-loan_emi
    if income and surplus<0:
        result.append({"title":"Recorded outflow is above income","detail":f"Recorded expenses and estimated loan payments exceed monthly income by ₹{abs(surplus)/100:,.2f}. Review the largest expense categories and upcoming loan payments."})
    elif income and loan_emi/income>=.4:
        result.append({"title":"Loan payments take a large share of income","detail":f"Recorded or estimated monthly loan payments are {loan_emi/income:.0%} of monthly income. Check the budget impact before taking on another repayment."})
    fund=p["emergency_fund_minor"] if p else 0
    if expenses>0 and fund<expenses*3:
        months=fund/expenses
        result.append({"title":"Build a cash reserve","detail":f"Your recorded emergency fund covers about {months:.1f} months of current recorded expenses. Consider a goal toward three months of expenses."})
    over=[b for b in budgets(user_id,month) if b["spent_minor"]>b["monthly_limit_minor"]]
    for b in over:
        result.append({"title":f"{b['category']} budget is over limit","detail":f"Recorded spending is ₹{(b['spent_minor']-b['monthly_limit_minor'])/100:,.2f} above this month's limit."})
    if not result:
        result.append({"title":"Keep your records current","detail":"Add transactions, loans, and savings goals to get guidance based on your current financial picture."})
    return result
