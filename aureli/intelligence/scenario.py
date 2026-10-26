"""Pure financial what-if calculations. This module never writes to persistence."""
from decimal import Decimal, ROUND_HALF_UP
from aureli.loans.service import loan_summary

def simulate_monthly_plan(*, income, expenses, existing_emi=0, income_change_pct=0,
                           expense_reduction_pct=0, spending_target_change=0,
                           new_loan_principal=0, new_loan_rate=0, new_loan_months=12,
                           extra_debt_payment=0, monthly_savings=0, initial_savings=0,
                           projection_months=12):
    vals=[Decimal(str(v)) for v in (income,expenses,existing_emi,income_change_pct,expense_reduction_pct,spending_target_change,new_loan_principal,new_loan_rate,extra_debt_payment,monthly_savings,initial_savings)]
    if any(not v.is_finite() for v in vals): raise ValueError("Scenario values must be finite numbers.")
    income,expenses,existing_emi,income_change_pct,expense_reduction_pct,spending_target_change,new_loan_principal,new_loan_rate,extra_debt_payment,monthly_savings,initial_savings=vals
    if income<0 or expenses<0 or existing_emi<0 or new_loan_principal<0 or extra_debt_payment<0 or monthly_savings<0 or initial_savings<0:
        raise ValueError("Income, expenses, debts, savings and payments cannot be negative.")
    if not 0<=expense_reduction_pct<=100: raise ValueError("Expense reduction must be between 0 and 100 percent.")
    months=int(projection_months)
    if months<1 or months>600: raise ValueError("Projection length must be from 1 to 600 months.")
    monthly_income=income*(1+income_change_pct/100)
    monthly_expenses=max(Decimal(0),expenses*(1-expense_reduction_pct/100)+spending_target_change)
    if new_loan_principal>0:
        if int(new_loan_months)<1: raise ValueError("New loan tenure must be positive.")
        new_loan=loan_summary(new_loan_principal,new_loan_rate,int(new_loan_months))
        new_emi=new_loan["emi"]
    else:
        new_loan=None; new_emi=Decimal(0)
    surplus=monthly_income-monthly_expenses-existing_emi-new_emi-extra_debt_payment-monthly_savings
    projected_savings=initial_savings+monthly_savings*months
    total_debt_service=existing_emi+new_emi+extra_debt_payment
    debt_to_income=(total_debt_service/monthly_income) if monthly_income>0 else None
    return {
        "monthly_income":monthly_income.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP),
        "monthly_expenses":monthly_expenses.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP),
        "new_loan_emi":new_emi,
        "monthly_surplus_after_plan":surplus.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP),
        "projected_savings":projected_savings.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP),
        "debt_to_income":debt_to_income,
        "projection_months":months,
        "new_loan_interest":new_loan["total_interest"] if new_loan else Decimal(0),
    }

def forecast_from_history(history, initial_savings=0, months=3):
    """Simple average of up to 3 latest recorded months; no external prediction."""
    entries=list(history)[-3:]
    if not entries: return {"months":0,"average_income":0,"average_expenses":0,"average_surplus":0,"projected_savings":Decimal(str(initial_savings))}
    avg=lambda key: sum((Decimal(str(row[key])) for _,row in entries),Decimal(0))/len(entries)
    income,expenses,surplus=avg("income"),avg("expenses"),avg("surplus")
    return {"months":len(entries),"average_income":income,"average_expenses":expenses,"average_surplus":surplus,
            "projected_savings":Decimal(str(initial_savings))+max(Decimal(0),surplus)*int(months)}
