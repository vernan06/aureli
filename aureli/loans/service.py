"""One authoritative fixed-rate monthly amortization implementation."""
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

CENT=Decimal("0.01")
def money(value): return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)

def amortization_schedule(principal, annual_rate, months, extra_payment=0):
    try:
        balance=money(principal); annual=Decimal(str(annual_rate)); month_value=Decimal(str(months)); extra=money(extra_payment)
    except (InvalidOperation, TypeError, ValueError) as exc: raise ValueError("Enter finite loan amounts, rate, and whole-month term.") from exc
    if not annual.is_finite() or not month_value.is_finite() or month_value!=month_value.to_integral_value(): raise ValueError("Interest rate must be finite and loan duration must be a whole number of months.")
    rate=annual/Decimal(1200); n=int(month_value)
    if balance <= 0 or n < 1 or n>600 or annual < 0 or annual>10000 or extra < 0: raise ValueError("Loan principal must be positive, duration 1–600 months, and rate and extra payment within supported non-negative ranges.")
    payment=money(balance/n) if rate==0 else money(balance*rate/(1-(1+rate)**(-n)))
    rows=[]
    for period in range(1,n+1):
        interest=money(balance*rate)
        planned_principal=money(payment-interest+extra)
        principal_paid=balance if period==n or planned_principal>=balance else planned_principal
        total=money(principal_paid+interest)
        balance=money(balance-principal_paid)
        rows.append({"period":period,"payment":total,"principal":principal_paid,"interest":interest,"balance":balance})
        if balance == 0: break
    return rows

def loan_summary(principal, annual_rate, months, extra_payment=0):
    rows=amortization_schedule(principal,annual_rate,months,extra_payment)
    return {"emi":rows[0]["payment"],"total_paid":sum((r["payment"] for r in rows),Decimal(0)),"total_interest":sum((r["interest"] for r in rows),Decimal(0)),"months":len(rows),"schedule":rows}

def save_loan(user_id, name, principal, rate, months, start_date, extra_payment=0):
    from aureli.finance.service import to_minor
    from aureli.persistence.db import connect
    if not name.strip(): raise ValueError("Loan name is required.")
    date.fromisoformat(start_date)
    try: annual=Decimal(str(rate)); month_value=Decimal(str(months)); extra_value=Decimal(str(extra_payment))
    except (InvalidOperation,TypeError,ValueError) as exc: raise ValueError("Enter a finite rate, non-negative extra amount, and whole-month term.") from exc
    if not annual.is_finite() or not month_value.is_finite() or not extra_value.is_finite() or annual<0 or annual>10000 or month_value<1 or month_value!=month_value.to_integral_value() or extra_value<0 or month_value>600: raise ValueError("Enter a rate from 0 to 10,000%, non-negative extra amount, and whole-month term from 1 to 600.")
    principal_minor=to_minor(principal)
    extra_minor=int((extra_value*100).quantize(Decimal("1"),rounding=ROUND_HALF_UP))
    with connect() as db:
        cur=db.execute("INSERT INTO loans(user_id,name,principal_minor,annual_rate,term_months,start_date,extra_payment_minor) VALUES(?,?,?,?,?,?,?)",(user_id,name.strip(),principal_minor,float(annual),int(month_value),start_date,extra_minor))
        return cur.lastrowid

def user_loans(user_id):
    from aureli.persistence.db import connect
    with connect() as db: return db.execute("SELECT * FROM loans WHERE user_id=? ORDER BY id DESC",(user_id,)).fetchall()

def loan_balance(user_id: int, loan_id: int):
    from aureli.persistence.db import connect
    with connect() as db:
        row=db.execute("SELECT l.principal_minor-COALESCE(SUM(p.principal_minor),0) balance_minor FROM loans l LEFT JOIN loan_payments p ON p.loan_id=l.id WHERE l.id=? AND l.user_id=? GROUP BY l.id",(loan_id,user_id)).fetchone()
    return max(0,row["balance_minor"]) if row else None

def record_loan_payment(user_id: int, loan_id: int, amount, principal_paid, paid_on: str, note="", payment_type="scheduled"):
    from aureli.finance.service import to_minor
    from aureli.persistence.db import connect
    date.fromisoformat(paid_on)
    try: principal_value=Decimal(str(principal_paid))
    except (InvalidOperation,TypeError,ValueError) as exc: raise ValueError("Principal amount must be finite.") from exc
    if not principal_value.is_finite() or principal_value<0: raise ValueError("Principal amount must be finite and non-negative.")
    total=to_minor(amount); principal=int((principal_value*100).quantize(Decimal("1"),rounding=ROUND_HALF_UP))
    if payment_type not in {"scheduled","extra_principal"}: raise ValueError("Invalid payment type.")
    if total < 100: raise ValueError("Enter the actual payment amount; values below ₹1.00 are too small to record.")
    if principal <= 0: raise ValueError("Enter the principal amount shown by your lender. Zero-principal placeholder payments cannot be recorded.")
    interest=0 if payment_type=="extra_principal" else total-principal
    if principal>total or interest<0: raise ValueError("Principal paid must be between zero and the payment amount.")
    if payment_type=="extra_principal" and principal!=total: raise ValueError("An extra-principal payment must be applied entirely to principal.")
    with connect() as db:
        loan=db.execute("SELECT principal_minor FROM loans WHERE id=? AND user_id=?",(loan_id,user_id)).fetchone()
        if not loan: raise ValueError("Loan was not found for this account.")
        paid=db.execute("SELECT COALESCE(SUM(principal_minor),0) FROM loan_payments WHERE loan_id=?",(loan_id,)).fetchone()[0]
        if principal>loan["principal_minor"]-paid: raise ValueError("Principal payment exceeds the recorded outstanding balance.")
        duplicate=db.execute("SELECT 1 FROM loan_payments WHERE loan_id=? AND amount_minor=? AND principal_minor=? AND paid_on=? AND note=? AND payment_type=? LIMIT 1",
            (loan_id,total,principal,paid_on,note.strip(),payment_type)).fetchone()
        if duplicate: raise ValueError("This exact payment is already recorded. Check the payment history before submitting again.")
        db.execute("INSERT INTO loan_payments(loan_id,amount_minor,paid_on,note,principal_minor,interest_minor,payment_type) VALUES(?,?,?,?,?,?,?)",(loan_id,total,paid_on,note.strip(),principal,interest,payment_type))

def loan_payment_history(user_id: int, loan_id: int):
    from aureli.persistence.db import connect
    with connect() as db:
        return db.execute("SELECT p.* FROM loan_payments p JOIN loans l ON l.id=p.loan_id WHERE l.user_id=? AND l.id=? ORDER BY p.paid_on DESC,p.id DESC",(user_id,loan_id)).fetchall()

def compare_debt_payoff(loans, extra_monthly, strategy="avalanche"):
    """Compare snowball/avalanche payoff using the shared monthly money rules."""
    if strategy not in {"snowball","avalanche"}: raise ValueError("Strategy must be snowball or avalanche.")
    extra=money(extra_monthly)
    if extra<0: raise ValueError("Extra payment cannot be negative.")
    debts=[]
    for row in loans:
        principal=Decimal(row["principal_minor"])/100
        if principal<=0: continue
        summary=loan_summary(principal,row["annual_rate"],row["term_months"],row["extra_payment_minor"]/100)
        regular=loan_summary(principal,row["annual_rate"],row["term_months"])["emi"]
        debts.append({"id":row["id"],"name":row["name"],"balance":money(principal),"rate":Decimal(str(row["annual_rate"])),"payment":regular,"months":row["term_months"],"extra_payment":Decimal(row["extra_payment_minor"])/100,"_baseline":summary})
    if strategy=="snowball": debts.sort(key=lambda d:(d["balance"],-d["rate"]))
    else: debts.sort(key=lambda d:(-d["rate"],d["balance"]))
    total_interest=Decimal(0); payoff={}; months=0
    while debts and months<1200:
        months+=1; extra_pool=extra
        for debt in debts:
            if debt["balance"]<=0: continue
            interest=money(debt["balance"]*debt["rate"]/1200)
            total_interest+=interest
            payment=min(debt["payment"],money(debt["balance"]+interest))
            debt["balance"]=money(max(Decimal(0),debt["balance"]-(payment-interest)))
            extra_pool+=debt["extra_payment"]
        for debt in debts:
            if debt["balance"]<=0: continue
            interest=money(debt["balance"]*debt["rate"]/1200)
            principal=min(debt["balance"],extra_pool)
            debt["balance"]=money(debt["balance"]-principal); extra_pool-=principal
            if debt["balance"]==0: payoff[debt["name"]]=months
        debts=[debt for debt in debts if debt["balance"]>0]
    return {"strategy":strategy,"months":months,"total_interest":total_interest,"payoff_months":payoff,"paid_off":not debts}
