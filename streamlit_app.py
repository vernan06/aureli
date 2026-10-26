from __future__ import annotations

import csv
import io
import secrets
from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import plotly.graph_objects as go
import streamlit as st

from aureli.auth.service import authenticate, register, delete_local_account
from aureli.finance.service import (add_goal, add_transaction, budgets, delete_transaction,
    goals, monthly_summary, profile, set_budget, set_profile, to_minor, transactions, update_transaction)
from aureli.loans.service import loan_summary, save_loan, user_loans
from aureli.intelligence.service import recommendations
from aureli.finance.service import (add_category, add_recurring_obligation, categories, recurring_obligations,
    set_recurring_active, contribute_to_goal, goals_with_progress, goal_contributions, update_goal, delete_goal,
    monthly_history, delete_budget, set_budget)
from aureli.loans.service import loan_balance, loan_payment_history, record_loan_payment, compare_debt_payoff
from aureli.intelligence.scenario import simulate_monthly_plan, forecast_from_history
from aureli.ml.inference import artifact_path, load_artifact, predict, explain_prediction
from aureli.persistence.db import initialize

MODEL_DIR=Path("artifacts/models")
st.set_page_config(page_title="Aureli — Financial intelligence", page_icon="✦", layout="wide")
st.markdown("""<style>
:root{color-scheme:dark;--ink:#edf7f3;--muted:#91aaa7;--panel:#10242d;--line:#244149;--mint:#58d6b1}
.stApp{background:radial-gradient(ellipse at 78% -12%,#153b3b 0%,#0b1d25 42%,#08151e 75%);color:var(--ink)}
[data-testid="stSidebar"]{background:#0b1c24;border-right:1px solid #1a3740}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:9px;padding:.36rem .5rem}
[data-testid="stMetric"]{background:linear-gradient(145deg,#122b33,#0e2029);border:1px solid var(--line);padding:17px 19px;border-radius:16px;box-shadow:0 8px 26px #0002}
[data-testid="stMetricLabel"]{color:var(--muted)} [data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
h1,h2,h3{color:#eaf5f0;letter-spacing:-.025em}.aureli-sub{color:var(--muted);font-size:1rem;margin-top:-.7rem}
.stButton>button,.stFormSubmitButton>button{border-radius:10px;border:1px solid #328b77;transition:all .16s ease}
.stButton>button:hover,.stFormSubmitButton>button:hover{border-color:var(--mint);color:var(--mint)}
[data-testid="stDataFrame"],[data-testid="stPlotlyChart"],[data-testid="stExpander"]{border-radius:14px;overflow:hidden}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#3bc6a5,#73e1bb)}
button:focus-visible,input:focus-visible{outline:2px solid var(--mint)!important;outline-offset:2px}
@media(max-width:700px){.block-container{padding:1.1rem .8rem 2rem}.stMetric{padding:12px!important}}
</style>""", unsafe_allow_html=True)
initialize()

def money(minor): return f"₹{minor/100:,.2f}"

def notify_success(message):
    st.session_state["aureli_notice"]=message
    st.rerun()

notice=st.session_state.pop("aureli_notice",None)
if notice: st.success(notice)

def login_screen():
    st.markdown("# ✦ Aureli")
    st.markdown('<p class="aureli-sub">Your personal financial intelligence.</p>', unsafe_allow_html=True)
    signin, signup = st.tabs(["Sign in", "Create account"])
    with signin:
        with st.form("signin"):
            email=st.text_input("Email"); password=st.text_input("Password",type="password")
            submit=st.form_submit_button("Sign in",use_container_width=True)
        if submit:
            user=authenticate(email,password)
            if user: st.session_state.user=user; st.rerun()
            st.error("Email or password is incorrect.")
    with signup:
        with st.form("signup"):
            name=st.text_input("Name"); email=st.text_input("Email address",key="new_email")
            password=st.text_input("Password (10+ characters)",type="password",key="new_password")
            submit=st.form_submit_button("Create account",use_container_width=True)
        if submit:
            try:
                user_id=register(email,password,name)
                st.session_state.user={"id":user_id,"name":name.strip()}; st.rerun()
            except Exception as exc: st.error(str(exc) if "UNIQUE" not in str(exc) else "That email already has an account.")
    st.divider()
    st.caption("Try Aureli without setting up an account. Guest records are scoped to this private browser session and cleared when you end it.")
    if st.button("Continue in guest demo mode",use_container_width=True):
        guest_id=register(f"guest-{uuid4().hex}@local.test",secrets.token_urlsafe(32),"Guest")
        st.session_state.user={"id":guest_id,"name":"Guest","demo":True}; st.rerun()

if "user" not in st.session_state:
    login_screen(); st.stop()

user=st.session_state.user; uid=int(user["id"])
with st.sidebar:
    st.markdown("## ✦ Aureli")
    st.caption("Your personal financial intelligence")
    section=st.radio("Navigate",["Overview","EMI Prediction","Model Laboratory","Loan Simulator","Financial Planner"],label_visibility="collapsed")
    if section=="Financial Planner":
        page=st.radio("Planner",["Transactions","Budgets","Savings Goals","Financial Twin","Reports","Settings"],label_visibility="collapsed")
    else:
        page={"Overview":"Dashboard","EMI Prediction":"AI Assessment","Model Laboratory":"Model Laboratory","Loan Simulator":"Loans & EMIs"}[section]
    st.divider(); st.caption("Guest demo session" if user.get("demo") else f"Signed in as {user['name']}")
    if st.button("End guest session" if user.get("demo") else "Sign out",use_container_width=True):
        if user.get("demo"): delete_local_account(uid)
        del st.session_state.user; st.rerun()

def page_header(title, subtitle):
    st.title(title); st.markdown(f'<p class="aureli-sub">{subtitle}</p>',unsafe_allow_html=True)

if page=="Dashboard":
    page_header("Your financial overview", "A clear view of the money you have recorded.")
    summary=monthly_summary(uid); loan_rows=user_loans(uid); goal_rows=goals_with_progress(uid)
    loan_total=sum(loan_balance(uid,r["id"]) or 0 for r in loan_rows)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Income this month",money(summary["income"])); c2.metric("Expenses this month",money(summary["expenses"]))
    c3.metric("Monthly surplus",money(summary["surplus"])); c4.metric("Loan principal recorded",money(loan_total))
    left,right=st.columns([1.3,1])
    with left:
        st.subheader("Cash flow")
        tx=transactions(uid)
        months={}
        for t in tx:
            m=t["occurred_on"][:7]; months.setdefault(m,{"income":0,"expense":0}); months[m][t["kind"]]+=t["amount_minor"]
        labels=sorted(months)[-6:]
        if labels:
            display_labels=[date.fromisoformat(f"{month}-01").strftime("%b %Y") for month in labels]
            fig=go.Figure()
            fig.add_bar(name="Income",x=display_labels,y=[months[m]["income"]/100 for m in labels],marker_color="#58d6b1")
            fig.add_bar(name="Expenses",x=display_labels,y=[months[m]["expense"]/100 for m in labels],marker_color="#e89581")
            fig.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#d9e9e3",yaxis_title="INR",legend_orientation="h",margin=dict(l=10,r=10,t=20,b=10),xaxis=dict(type="category",title=None),yaxis=dict(tickprefix="₹",gridcolor="#1a343c"),font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Your cash-flow story will appear here after you record income or expenses.")
    with right:
        st.subheader("Savings goals")
        if goal_rows:
            for g in goal_rows:
                ratio=min(g["total_saved_minor"]/g["target_minor"],1)
                st.write(f"**{g['name']}** · {money(g['total_saved_minor'])} of {money(g['target_minor'])}")
                st.progress(ratio)
        else: st.info("Add a savings goal to track progress here.")
        st.subheader("Budget watch")
        rows=budgets(uid)
        if rows:
            for b in rows:
                ratio=b['spent_minor']/b['monthly_limit_minor']
                remaining=b['monthly_limit_minor']-b['spent_minor']
                st.write(f"**{b['category']}** · {money(b['spent_minor'])} / {money(b['monthly_limit_minor'])}")
                st.caption((f"{money(remaining)} remaining" if remaining>=0 else f"Over budget by {money(-remaining)}")+f" · {ratio:.0%} used")
                st.progress(min(ratio,1.0))
        else: st.caption("No budgets recorded yet.")
    st.subheader("Personal guidance")
    for item in recommendations(uid):
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.write(item["detail"])
    st.subheader("EMI intelligence")
    eligibility_ready=artifact_path("emi_eligibility").is_file()
    emi_ready=artifact_path("max_monthly_emi").is_file()
    a,b=st.columns(2)
    a.metric("Eligibility model","Ready" if eligibility_ready else "Awaiting authorized training data")
    b.metric("Affordable EMI model","Ready" if emi_ready else "Awaiting authorized training data")
    if not eligibility_ready or not emi_ready:
        st.caption("Model estimates appear when their saved, authorized pipelines are available.")

elif page=="Transactions":
    page_header("Transactions","Record income and expenses in one ledger.")
    with st.form("transaction_form",clear_on_submit=True):
        a,b,c=st.columns(3); kind=a.selectbox("Type",["expense","income"]); amount=b.number_input("Amount (₹)",min_value=0.01,step=100.0); category=c.text_input("Category",placeholder="Groceries")
        description=st.text_input("Description"); d,e=st.columns(2); occurred=d.date_input("Date",value=date.today()); recurring=e.checkbox("Recurring")
        submitted=st.form_submit_button("Add transaction")
    if submitted:
        try: add_transaction(uid,kind,amount,category,description,occurred.isoformat(),recurring); notify_success("Transaction saved.")
        except ValueError as exc: st.error(str(exc))
    rows=transactions(uid)
    if rows:
        st.subheader("Recent activity")
        st.dataframe([{"Date":r["occurred_on"],"Type":r["kind"].title(),"Category":r["category"],"Description":r["description"],"Amount":money(r["amount_minor"]),"Recurring":"Yes" if r["recurring"] else "No"} for r in rows],use_container_width=True,hide_index=True)
        with st.expander("Edit a transaction"):
            rows_by_id={int(r["id"]):r for r in rows}
            selected_id=st.selectbox("Transaction",list(rows_by_id),format_func=lambda key:f"#{key} · {rows_by_id[key]['occurred_on']} · {rows_by_id[key]['kind']} · {rows_by_id[key]['category']}",key="edit_tx_select")
            selected=rows_by_id[selected_id]
            with st.form("edit_transaction"):
                a,b,c=st.columns(3); edit_kind=a.selectbox("Type",["expense","income"],index=["expense","income"].index(selected["kind"])); edit_amount=b.number_input("Amount (₹)",min_value=.01,value=selected["amount_minor"]/100); edit_category=c.text_input("Category",value=selected["category"])
                edit_description=st.text_input("Description",value=selected["description"]); d,e=st.columns(2); edit_date=d.date_input("Date",value=date.fromisoformat(selected["occurred_on"])); edit_recurring=e.checkbox("Recurring",value=bool(selected["recurring"]))
                if st.form_submit_button("Save changes"):
                    try: update_transaction(uid,selected["id"],kind=edit_kind,amount=edit_amount,category=edit_category,description=edit_description,occurred_on=edit_date.isoformat(),recurring=edit_recurring); notify_success("Transaction updated.")
                    except ValueError as exc: st.error(str(exc))
        with st.form("delete_transaction"):
            delete_ids={int(r["id"]):r for r in rows}
            chosen=st.selectbox("Choose a transaction",list(delete_ids),format_func=lambda key:f"{delete_ids[key]['occurred_on']} · {delete_ids[key]['kind']} · {delete_ids[key]['category']} · {money(delete_ids[key]['amount_minor'])}",key="delete_tx_select")
            confirm=st.checkbox("I understand this permanently removes the selected transaction")
            if st.form_submit_button("Delete selected transaction"):
                if confirm and delete_transaction(uid,chosen): notify_success("Transaction removed.")
                else: st.error("Confirm the deletion to continue.")
    else: st.info("No transactions yet. Add your first income or expense above.")

elif page=="Budgets":
    page_header("Budgets","Set monthly category limits and see spending against this month’s recorded transactions.")
    with st.form("budget_form"):
        a,b=st.columns(2); category=a.text_input("Category"); amount=b.number_input("Monthly limit (₹)",min_value=0.01,step=500.0)
        if st.form_submit_button("Save budget"):
            try: set_budget(uid,category,amount); notify_success("Budget saved.")
            except ValueError as exc: st.error(str(exc))
    rows=budgets(uid)
    if rows:
        for row in rows:
            spent=row["spent_minor"]; limit=row["monthly_limit_minor"]; remaining=limit-spent; ratio=spent/limit
            with st.container(border=True):
                left,right=st.columns([3,1]); left.markdown(f"**{row['category']}**"); right.markdown(f"**{ratio:.0%} used**")
                st.progress(min(ratio,1.0))
                st.caption(f"Spent {money(spent)} of {money(limit)} · "+(f"{money(remaining)} remaining" if remaining>=0 else f"overspent by {money(-remaining)}"))
    else: st.info("Create a budget to start tracking category spending.")
    if rows:
        with st.form("budget_delete"):
            category=st.selectbox("Budget to remove",[r["category"] for r in rows])
            if st.form_submit_button("Remove budget"):
                delete_budget(uid,category); notify_success("Budget removed.")

elif page=="Loans & EMIs":
    page_header("Loans & EMIs","Save loans and use the same amortization service for payment and interest estimates.")
    with st.container(border=True):
        st.subheader("Plan a new loan")
        a,b,c=st.columns(3); name=a.text_input("Loan name",placeholder="Home, vehicle, education…",key="new_loan_name"); principal=b.number_input("Principal (₹)",min_value=1.0,value=100000.0,step=10000.0,key="new_loan_principal"); rate=c.number_input("Annual interest (%)",min_value=0.0,value=12.0,step=.25,key="new_loan_rate")
        d,e,f=st.columns(3); months=d.number_input("Duration (months)",min_value=1,max_value=600,value=12,key="new_loan_months"); start=e.date_input("Start date",value=date.today(),key="new_loan_start"); extra=f.number_input("Optional extra principal/month (₹)",min_value=0.0,value=0.0,step=500.0,key="new_loan_extra")
        preview=loan_summary(principal,rate,months,extra)
        p1,p2,p3,p4=st.columns(4); p1.metric("Estimated monthly payment",f"₹{preview['emi']:,.2f}"); p2.metric("Total repayment",f"₹{preview['total_paid']:,.2f}"); p3.metric("Total interest",f"₹{preview['total_interest']:,.2f}"); p4.metric("Payoff duration",f"{preview['months']} months")
        st.caption("EMI is your estimated fixed monthly installment. Total interest is the estimated borrowing cost over this schedule. Actual lender fees, dates and rounding may differ.")
        if st.button("Save this loan",type="primary",use_container_width=False):
            if not name.strip(): st.error("Add a name so you can recognize this loan later.")
            else:
                try: save_loan(uid,name,principal,rate,months,start.isoformat(),extra); notify_success("Loan saved. Review the schedule below.")
                except ValueError as exc: st.error(str(exc))
    rows=user_loans(uid)
    if rows:
        for loan in rows:
            with st.expander(f"{loan['name']} · {money(loan['principal_minor'])}"):
                outstanding=loan_balance(uid,loan["id"])
                start_date=date.fromisoformat(loan["start_date"])
                elapsed=max(0,(date.today().year-start_date.year)*12+date.today().month-start_date.month)
                remaining_term=max(1,int(loan["term_months"])-elapsed)
                result=loan_summary(Decimal(outstanding)/100,loan["annual_rate"],remaining_term,Decimal(loan["extra_payment_minor"])/100) if outstanding else {"emi":Decimal(0),"total_interest":Decimal(0),"months":0,"schedule":[]}
                a,b,c=st.columns(3); a.metric("Estimated monthly payment",f"₹{result['emi']:,.2f}"); b.metric("Estimated remaining interest",f"₹{result['total_interest']:,.2f}"); c.metric("Estimated payoff time",f"{result['months']} months")
                progress=1-outstanding/max(1,loan["principal_minor"])
                st.progress(max(0.0,min(progress,1.0)))
                st.caption(f"Recorded principal outstanding: {money(outstanding)} · initial principal: {money(loan['principal_minor'])} · estimated remaining term: {remaining_term} months. Payment estimates use the original contract rate and remaining term; confirm actual figures with your lender.")
                st.dataframe([{"Month":x["period"],"Payment":f"₹{x['payment']:,.2f}","Principal":f"₹{x['principal']:,.2f}","Interest":f"₹{x['interest']:,.2f}","Balance":f"₹{x['balance']:,.2f}"} for x in result["schedule"]],use_container_width=True,hide_index=True)
                if result["schedule"]:
                    composition=go.Figure(); composition.add_bar(name="Principal",x=[f"Month {x['period']}" for x in result["schedule"]],y=[float(x["principal"]) for x in result["schedule"]],marker_color="#58d6b1"); composition.add_bar(name="Interest",x=[f"Month {x['period']}" for x in result["schedule"]],y=[float(x["interest"]) for x in result["schedule"]],marker_color="#e89581"); composition.update_layout(barmode="stack",title="Payment breakdown",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#d9e9e3",yaxis_title="INR",margin=dict(l=10,r=10,t=40,b=10),yaxis=dict(tickprefix="₹",gridcolor="#1a343c")); st.plotly_chart(composition,use_container_width=True)
                payment_version=st.session_state.get(f"pay_version_{loan['id']}",0)
                payment_type=st.selectbox("Payment type",["scheduled","extra_principal"],format_func=lambda x:"Scheduled payment" if x=="scheduled" else "Extra principal payment",key=f"pay_type_{loan['id']}")
                if payment_type=="scheduled":
                    default_total=float(result["schedule"][0]["payment"]) if result["schedule"] else 1.0
                    default_principal=float(result["schedule"][0]["principal"]) if result["schedule"] else 1.0
                    payment_help="Use the total and principal amounts shown on your lender statement. Aureli pre-fills an estimate for convenience."
                else:
                    default_total=1000.0; default_principal=default_total
                    payment_help="Extra principal payments reduce balance. Confirm the amount your lender applies to principal."
                st.caption(payment_help)
                with st.form(f"payment_{loan['id']}_{payment_type}_{payment_version}"):
                    pa,pb,pc=st.columns(3)
                    total_paid=pa.number_input("Total paid (₹)",min_value=0.01,value=max(default_total,.01),step=500.0,key=f"pay_total_{loan['id']}_{payment_type}_{payment_version}")
                    principal_paid=pb.number_input("Principal component (₹)",min_value=0.0,max_value=float(total_paid),value=min(default_principal,float(total_paid)),step=500.0,key=f"pay_principal_{loan['id']}_{payment_type}_{payment_version}")
                    paid_date=pc.date_input("Paid on",value=date.today(),key=f"pay_date_{loan['id']}_{payment_type}_{payment_version}")
                    note=st.text_input("Lender reference or note",key=f"pay_note_{loan['id']}_{payment_type}_{payment_version}")
                    st.caption(f"Expected outstanding after this payment: {money(max(0,outstanding-int(Decimal(str(principal_paid))*100)))}")
                    if st.form_submit_button("Record payment",type="primary"):
                        try:
                            record_loan_payment(uid,loan["id"],total_paid,principal_paid,paid_date.isoformat(),note,payment_type)
                            st.session_state[f"pay_version_{loan['id']}"]=payment_version+1
                            notify_success(f"{('Scheduled' if payment_type=='scheduled' else 'Extra principal')} payment recorded successfully.")
                        except ValueError as exc: st.error(str(exc))
                history=loan_payment_history(uid,loan["id"])
                if history: st.dataframe([{"Date":x["paid_on"],"Type":x["payment_type"],"Total":money(x["amount_minor"]),"Principal":money(x["principal_minor"]),"Interest":money(x["interest_minor"]),"Note":x["note"]} for x in history],use_container_width=True,hide_index=True)
    else: st.info("Add a loan to see its payment schedule.")
    with st.expander("Compare a repayment scenario",expanded=bool(rows)):
        labels={f"{r['name']} · {money(loan_balance(uid,r['id']))} outstanding · {r['annual_rate']}%":r for r in rows if loan_balance(uid,r['id'])>0}
        choices=["Custom scenario",*labels]
        selected=st.selectbox("Baseline loan",choices,key="scenario_baseline")
        if selected!="Custom scenario":
            baseline_loan=labels[selected]; base_principal=loan_balance(uid,baseline_loan["id"])/100
            base_rate=float(baseline_loan["annual_rate"]); base_months=int(baseline_loan["term_months"])
        else: baseline_loan=None; base_principal=500000.0; base_rate=9.0; base_months=60
        col1,col2=st.columns(2)
        scenario_key="custom" if baseline_loan is None else str(baseline_loan["id"])
        p=col1.number_input("Baseline principal (₹)",min_value=1.0,value=float(base_principal),step=10000.0,key=f"scenario_principal_{scenario_key}")
        r=col2.number_input("Annual interest (%)",min_value=0.0,value=float(base_rate),step=.25,key=f"scenario_rate_{scenario_key}")
        n=col1.number_input("Remaining term (months)",min_value=1,max_value=600,value=max(1,min(600,base_months)),key=f"scenario_months_{scenario_key}")
        x=col2.number_input("Additional principal per month (₹)",min_value=0.0,step=500.0,key=f"scenario_extra_{scenario_key}")
        before=loan_summary(p,r,n); after=loan_summary(p,r,n,x)
        a,b,c=st.columns(3); a.metric("Interest saved",f"₹{before['total_interest']-after['total_interest']:,.2f}"); b.metric("Months saved",f"{before['months']-after['months']}"); c.metric("New payoff time",f"{after['months']} months")
        fig=go.Figure(); fig.add_bar(name="Baseline interest",x=["Current plan"],y=[float(before["total_interest"])]); fig.add_bar(name="With extra principal",x=["Accelerated plan"],y=[float(after["total_interest"])]); fig.update_layout(title="Estimated total interest",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#d9e9e3",yaxis_title="INR",showlegend=False,margin=dict(l=10,r=10,t=40,b=10),yaxis=dict(tickprefix="₹",gridcolor="#1a343c")); st.plotly_chart(fig,use_container_width=True)
        st.caption("This comparison is hypothetical and does not change loan balances or payment history.")
    if len(rows)>1:
        st.subheader("Debt strategies")
        extra_budget=st.number_input("Extra amount available across loans each month (₹)",min_value=0.0,step=500.0,key="strategy_extra")
        strategy_rows=[dict(loan)|{"principal_minor":loan_balance(uid,loan["id"])} for loan in rows]
        left,right=st.columns(2)
        for column,strategy,title in [(left,"snowball","Snowball · smallest balance first"),(right,"avalanche","Avalanche · highest rate first")]:
            result=compare_debt_payoff(strategy_rows,extra_budget,strategy)
            column.markdown(f"**{title}**")
            column.write(f"{result['months']} months · estimated interest ₹{result['total_interest']:,.2f}")
            column.caption("Simulation only; it does not change stored payments or loans.")

elif page=="Savings Goals":
    page_header("Savings goals","Keep goals connected to your financial plan.")
    with st.form("goal_form"):
        a,b,c=st.columns(3); name=a.text_input("Goal name"); target=b.number_input("Target (₹)",min_value=1.0,step=1000.0); saved=c.number_input("Already saved (₹)",min_value=0.0,step=500.0)
        due=st.date_input("Target date",value=None)
        if st.form_submit_button("Create goal"):
            if not name.strip(): st.error("Goal name is required.")
            else: add_goal(uid,name,target,saved,due.isoformat() if due else None); notify_success("Goal created.")
    rows=goals_with_progress(uid)
    if rows:
        for g in rows:
            st.write(f"**{g['name']}** — {money(g['total_saved_minor'])} / {money(g['target_minor'])}"+(f" · due {g['due_on']}" if g["due_on"] else ""))
            st.progress(min(g["total_saved_minor"]/g["target_minor"],1))
            with st.expander(f"Manage {g['name']}"):
                with st.form(f"contribute_{g['id']}"):
                    amount=st.number_input("Contribution (₹)",min_value=0.01,step=500.0,key=f"goal_amt_{g['id']}")
                    contributed=st.date_input("Date",value=date.today(),key=f"goal_date_{g['id']}")
                    note=st.text_input("Note",key=f"goal_note_{g['id']}")
                    if st.form_submit_button("Record contribution"):
                        try: contribute_to_goal(uid,g["id"],amount,contributed.isoformat(),note); notify_success("Contribution saved.")
                        except ValueError as exc: st.error(str(exc))
                history=goal_contributions(uid,g["id"])
                if history: st.dataframe([{"Date":x["contributed_on"],"Amount":money(x["amount_minor"]),"Note":x["note"]} for x in history],use_container_width=True,hide_index=True)
                with st.form(f"goal_edit_{g['id']}"):
                    new_name=st.text_input("Goal name",value=g["name"],key=f"goal_name_{g['id']}")
                    new_target=st.number_input("Target (₹)",min_value=1.0,value=g["target_minor"]/100,key=f"goal_target_{g['id']}")
                    if st.form_submit_button("Update goal"):
                        try: update_goal(uid,g["id"],name=new_name,target=new_target,due_on=g["due_on"]); notify_success("Goal updated.")
                        except ValueError as exc: st.error(str(exc))
                if st.button("Delete goal",key=f"goal_delete_{g['id']}"):
                    if delete_goal(uid,g["id"]): notify_success("Goal deleted.")
    else: st.info("No goals yet. Set a target and optional date to get started.")

elif page=="AI Assessment":
    page_header("EMI prediction","Use trained Aureli models for eligibility and affordable monthly EMI estimates.")
    st.warning("Predictions are advisory estimates only. They are not credit scores, lending decisions, or guaranteed approvals.")
    targets=[("emi_eligibility","EMI eligibility"),("max_monthly_emi","Maximum affordable EMI")]
    for target,title in targets:
        if not artifact_path(target).is_file():
            st.info(f"{title} is unavailable because its trained Aureli model artifact is not present. No prediction is generated until authorized local training produces it.")
            continue
        try: artifact=load_artifact(target)
        except Exception as exc: st.error(f"Could not load {title} artifact: {exc}"); continue
        with st.container(border=True):
            st.subheader(title)
            feature_columns=artifact.get("input_feature_columns",artifact["feature_columns"])
            groups={"Income & work":[],"Loan request":[],"Monthly commitments":[],"Credit & reserves":[],"Personal profile":[]}
            for feature in feature_columns:
                key=feature.casefold()
                if any(term in key for term in ("salary","income","employment","occupation","job")): group="Income & work"
                elif any(term in key for term in ("requested","loan_amount","loan_term","tenure","interest_rate")): group="Loan request"
                elif any(term in key for term in ("expense","emi","rent","groceries","travel","utility","bill","spending")): group="Monthly commitments"
                elif any(term in key for term in ("credit","bank_balance","bank_balance","savings","fund","account")): group="Credit & reserves"
                else: group="Personal profile"
                groups[group].append(feature)
            st.caption("Enter the details you know. Blank fields use values imputed by the saved training pipeline; review every field before interpreting a result.")
            with st.form(f"predict_{target}"):
                values={}; sections=st.tabs(list(groups))
                for section,group_name in zip(sections,groups):
                    with section:
                        for index,feature in enumerate(groups[group_name]):
                            numeric=feature in artifact.get("input_numeric_features",artifact["numeric_features"])
                            label=feature.replace("_"," ").strip().title()
                            help_text={"monthly_salary":"Your gross or take-home monthly income, matching the dataset definition.","requested_amount":"The amount you are considering borrowing.","credit_score":"Enter your reported score if known; do not guess.","monthly_rent":"Your monthly rent or housing cost.","bank_balance":"Current bank balance, in rupees.","current_emi_amount":"Total existing monthly loan installments."}.get(feature)
                            if numeric:
                                raw=section.text_input(label,placeholder="Optional · uses model preprocessing if blank",key=f"input_{target}_{feature}",help=help_text)
                                values[feature]=raw if raw.strip() else None
                            else:
                                options=artifact.get("input_categorical_values",{}).get(feature,[])
                                if options and len(options)<=50:
                                    selected=section.selectbox(label,["(Use training default)",*options],key=f"input_{target}_{feature}",help=help_text)
                                    values[feature]=None if selected=="(Use training default)" else selected
                                else:
                                    raw=section.text_input(label,placeholder="Optional · uses model preprocessing if blank",key=f"input_{target}_{feature}",help=help_text)
                                    values[feature]=raw if raw.strip() else None
                submitted=st.form_submit_button("Predict with saved Aureli model",type="primary")
            if submitted:
                if not any(value not in (None,"") for value in values.values()): st.error("Enter at least one financial detail before predicting.")
                else:
                    try:
                        result=predict(target,values)
                        if target=="emi_eligibility":
                            st.metric("Model assessment",result["prediction"])
                            if result["probabilities"]: st.dataframe([{"Class":label,"Model probability":f"{chance:.1%}"} for label,chance in result["probabilities"].items()],hide_index=True)
                            st.caption("Probabilities are model outputs and are not necessarily calibrated likelihoods of approval.")
                        else:
                            st.metric("Estimated maximum monthly EMI",f"₹{result['prediction']:,.2f}")
                            if result.get("range_limited"):
                                st.caption(f"Estimate was bounded to the range observed in training: ₹{result['observed_target_range'][0]:,.2f}–₹{result['observed_target_range'][1]:,.2f}.")
                        st.caption(f"Model: {result['model_name']} · dataset fingerprint: {result['dataset_sha256'][:12]}")
                        explanation=explain_prediction(target,values)
                        st.markdown(f"**Why this estimate?** {explanation['method']}")
                        if explanation["features"]: st.dataframe(explanation["features"],hide_index=True)
                        st.info("Use this as an educational estimate only. It does not assess your full financial circumstances or make a credit decision.")
                    except Exception as exc: st.error(f"Prediction could not be produced: {exc}")

elif page=="Model Laboratory":
    page_header("Model laboratory","Explore trained Aureli experiments and evaluation artifacts.")
    comparison=MODEL_DIR/"model_comparison.csv"
    if comparison.is_file():
        import pandas as pd
        table=pd.read_csv(comparison)
        st.subheader("Six-model comparison")
        classification=table[table["task"]=="classification"]
        regression=table[table["task"]=="regression"]
        model_tabs=st.tabs(["Eligibility classification","Maximum EMI regression"])
        with model_tabs[0]:
            columns={"model":"Model","accuracy":"Test accuracy","macro_f1":"Macro F1","weighted_f1":"Weighted F1","roc_auc_ovr_macro":"Macro ROC AUC","multiclass_brier_score":"Brier score","log_loss":"Log loss","expected_calibration_error_10_bins":"Calibration error"}
            st.dataframe(classification[list(columns)].rename(columns=columns).round(4),use_container_width=True,hide_index=True)
        with model_tabs[1]:
            columns={"model":"Model","mae":"Test MAE (₹)","rmse":"Test RMSE (₹)","r2":"Test R²"}
            st.dataframe(regression[list(columns)].rename(columns=columns).round(3),use_container_width=True,hide_index=True)
        st.caption("Metrics are from an untouched held-out test set. Classification selection is by validation macro F1; regression selection is by validation RMSE.")
        candidates=table[table.get("evaluation_path",pd.Series(index=table.index,dtype=object)).notna()] if "evaluation_path" in table else pd.DataFrame()
        if not candidates.empty:
            selected=st.selectbox("Evaluation artifact",candidates.to_dict("records"),format_func=lambda r:f"{r['target']} · {r['model']}")
            evaluation=Path(str(selected["evaluation_path"]))
            filenames=["confusion_matrix.html","calibration.html"] if selected["task"]=="classification" else ["actual_vs_predicted.html","residuals.html"]
            for filename in filenames:
                plot_path=evaluation/filename
                if plot_path.is_file():
                    st.markdown(f"**{filename.replace('_',' ').replace('.html','').title()}**")
                    html=plot_path.read_text(encoding="utf-8")
                    theme_script="""<script>(function(){function applyAureliTheme(){if(!window.Plotly)return;var plots=document.querySelectorAll('.js-plotly-plot');if(!plots.length){setTimeout(applyAureliTheme,80);return;}plots.forEach(function(plot){Plotly.relayout(plot,{'paper_bgcolor':'rgba(0,0,0,0)','plot_bgcolor':'#10242d','font.color':'#d9e9e3','xaxis.gridcolor':'#244149','yaxis.gridcolor':'#244149','xaxis.zerolinecolor':'#466169','yaxis.zerolinecolor':'#466169'});});}setTimeout(applyAureliTheme,200);})();</script>"""
                    st.components.v1.html(html.replace("</body>",theme_script+"</body>"),height=500,scrolling=True)
    else:
        st.info("No Aureli experiment results have been recorded yet. The authorized company dataset is available locally; run the documented six-model training command to create the experiments and artifacts.")
    for target,title in [("emi_eligibility","Classification"),("max_monthly_emi","Regression")]:
        metadata=MODEL_DIR/f"{target}.metadata.json"
        if not metadata.is_file(): continue
        import json
        info=json.loads(metadata.read_text(encoding="utf-8"))
        st.subheader(f"{title} champion · {info['model_name']} · version {info['registered_version']}")
        st.json(info["metrics"])
        if target=="emi_eligibility" and "confusion_matrix" in info["metrics"]:
            import plotly.express as px
            fig=px.imshow(info["metrics"]["confusion_matrix"],x=info["metrics"]["classes"],y=info["metrics"]["classes"],text_auto=True,color_continuous_scale="Teal",labels={"x":"Predicted","y":"Actual","color":"Count"})
            st.plotly_chart(fig,use_container_width=True)
        st.dataframe(info.get("feature_importance",[])[:20],hide_index=True)

elif page=="Financial Twin":
    page_header("Financial twin","Model a monthly plan using recorded cash flow. Scenarios never change saved records.")
    summary=monthly_summary(uid); loans=user_loans(uid); total_emi=0
    for loan in loans:
        outstanding=loan_balance(uid,loan["id"])
        if outstanding:
            total_emi+=int(loan_summary(Decimal(outstanding)/100,loan["annual_rate"],loan["term_months"],Decimal(loan["extra_payment_minor"])/100)["emi"]*100)
    p=profile(uid); income=summary["income"] or (p["monthly_income_minor"] if p else 0)
    savings_minor=(p["emergency_fund_minor"] if p else 0)+sum(g["total_saved_minor"] for g in goals_with_progress(uid))
    left,right=st.columns(2)
    with left:
        salary_delta=st.slider("Monthly income change (%)",-50,100,0,step=5)
        expense_reduction=st.slider("Monthly expense reduction (%)",0,50,0,step=5)
        spending_adjustment=st.number_input("Other monthly spending target change (₹; negative lowers spending)",value=0.0,step=500.0)
        new_principal=st.number_input("New loan scenario principal (₹)",min_value=0.0,step=10000.0)
        new_rate=st.number_input("New loan annual rate (%)",min_value=0.0,step=.25)
        new_tenure=st.number_input("New loan tenure (months)",min_value=1,max_value=600,value=60)
        extra=st.number_input("Extra monthly debt repayment (₹)",min_value=0.0,step=500.0)
        monthly_saving=st.number_input("Planned monthly savings (₹)",min_value=0.0,step=500.0)
        projection=st.selectbox("Projection horizon (months)",[3,6,12,24,36],index=2)
    plan=simulate_monthly_plan(income=income/100,expenses=summary["expenses"]/100,existing_emi=total_emi/100,
        income_change_pct=salary_delta,expense_reduction_pct=expense_reduction,spending_target_change=spending_adjustment,
        new_loan_principal=new_principal,new_loan_rate=new_rate,new_loan_months=new_tenure,
        extra_debt_payment=extra,monthly_savings=monthly_saving,initial_savings=savings_minor/100,projection_months=projection)
    recurring_total=sum(item["amount_minor"] for item in recurring_obligations(uid))
    forecast=forecast_from_history(monthly_history(uid),Decimal(savings_minor)/100,projection)
    with right:
        st.metric("Monthly surplus after scenario",f"₹{plan['monthly_surplus_after_plan']:,.2f}")
        st.metric("New loan estimated EMI",f"₹{plan['new_loan_emi']:,.2f}")
        st.metric("Projected savings",f"₹{plan['projected_savings']:,.2f}",f"{projection} months")
        st.metric("Debt service / income",f"{plan['debt_to_income']:.1%}" if plan["debt_to_income"] is not None else "Unavailable")
        st.caption(f"This is a planning estimate, not a credit score or lending decision. Recorded loan payment estimate: {money(total_emi)}.")
        st.metric("Tracked recurring obligations",money(recurring_total))
        st.subheader("Recent cash-flow average")
        st.write(f"Based on {forecast['months']} months recorded: average monthly income {money(int(forecast['average_income']))}, expenses {money(int(forecast['average_expenses']))}, surplus {money(int(forecast['average_surplus']))}.")
    st.info("The monthly average is descriptive, not a forecast guarantee. Add actual transactions and update your profile to improve the scenario baseline.")

elif page=="Reports":
    page_header("Reports","Download a portable snapshot of your saved financial records.")
    summary=monthly_summary(uid); tx=transactions(uid); budget_rows=budgets(uid); goal_rows=goals_with_progress(uid)
    loan_rows=user_loans(uid); obligations=recurring_obligations(uid,active_only=False)
    b=io.StringIO(); writer=csv.writer(b)
    writer.writerow(["Aureli financial report",date.today().isoformat()]); writer.writerow(["Metric","Amount (INR)"])
    writer.writerow(["Income this month",f"{summary['income']/100:.2f}"]); writer.writerow(["Expenses this month",f"{summary['expenses']/100:.2f}"]); writer.writerow(["Surplus this month",f"{summary['surplus']/100:.2f}"])
    writer.writerow([]); writer.writerow(["Transactions"]); writer.writerow(["Date","Type","Category","Description","Amount INR","Recurring"])
    for r in tx: writer.writerow([r["occurred_on"],r["kind"],r["category"],r["description"],f"{r['amount_minor']/100:.2f}",bool(r["recurring"])])
    writer.writerow([]); writer.writerow(["Budgets"]); writer.writerow(["Category","Monthly limit INR","Spent this month INR"])
    for r in budget_rows: writer.writerow([r["category"],f"{r['monthly_limit_minor']/100:.2f}",f"{r['spent_minor']/100:.2f}"])
    writer.writerow([]); writer.writerow(["Savings goals"]); writer.writerow(["Name","Target INR","Saved INR","Due date"])
    for r in goal_rows: writer.writerow([r["name"],f"{r['target_minor']/100:.2f}",f"{r['total_saved_minor']/100:.2f}",r["due_on"] or ""])
    writer.writerow([]); writer.writerow(["Loans"]); writer.writerow(["Name","Initial principal INR","Outstanding principal INR","Annual rate percent","Term months"])
    for r in loan_rows: writer.writerow([r["name"],f"{r['principal_minor']/100:.2f}",f"{loan_balance(uid,r['id'])/100:.2f}",r["annual_rate"],r["term_months"]])
    writer.writerow([]); writer.writerow(["Loan payments"]); writer.writerow(["Loan","Date","Payment type","Amount INR","Principal INR","Interest INR","Note"])
    for loan in loan_rows:
        for payment in loan_payment_history(uid,loan["id"]): writer.writerow([loan["name"],payment["paid_on"],payment["payment_type"],f"{payment['amount_minor']/100:.2f}",f"{payment['principal_minor']/100:.2f}",f"{payment['interest_minor']/100:.2f}",payment["note"]])
    writer.writerow([]); writer.writerow(["Recurring obligations"]); writer.writerow(["Name","Category","Monthly amount INR","Due day","Active"])
    for r in obligations: writer.writerow([r["name"],r["category"],f"{r['amount_minor']/100:.2f}",r["due_day"],bool(r["active"])])
    st.download_button("Download CSV report",b.getvalue(),file_name=f"aureli-report-{date.today().isoformat()}.csv",mime="text/csv")
    st.caption(f"Includes {len(tx)} transactions, {len(budget_rows)} budgets, {len(goal_rows)} goals, {len(loan_rows)} loans, and {len(obligations)} recurring obligations. Amounts come from your account data.")

else:
    page_header("Settings","Profile and account preferences.")
    p=profile(uid)
    with st.form("profile_form"):
        income=st.number_input("Typical monthly income (₹)",min_value=0.0,value=p["monthly_income_minor"]/100,step=1000.0)
        fund=st.number_input("Emergency fund (₹)",min_value=0.0,value=p["emergency_fund_minor"]/100,step=1000.0)
        if st.form_submit_button("Save profile"):
            set_profile(uid,income,fund); st.success("Profile saved.")
    st.subheader("Categories")
    with st.form("category_form"):
        name=st.text_input("Category name"); kind=st.selectbox("Category type",["expense","income"])
        if st.form_submit_button("Add category"):
            try: add_category(uid,name,kind); notify_success("Category saved.")
            except ValueError as exc: st.error(str(exc))
    cats=categories(uid)
    if cats: st.dataframe([{"Name":x["name"],"Type":x["kind"]} for x in cats],use_container_width=True,hide_index=True)
    st.subheader("Recurring obligations")
    with st.form("obligation_form"):
        a,b,c=st.columns(3); obligation=a.text_input("Name"); cat=b.text_input("Category"); amount=c.number_input("Monthly amount (₹)",min_value=.01,step=100.0)
        due_day=st.number_input("Due day",min_value=1,max_value=31,value=1)
        if st.form_submit_button("Add recurring obligation"):
            try: add_recurring_obligation(uid,obligation,cat,amount,due_day); notify_success("Obligation saved.")
            except ValueError as exc: st.error(str(exc))
    obligations=recurring_obligations(uid,active_only=False)
    if obligations:
        for item in obligations:
            c1,c2=st.columns([4,1]); c1.write(f"{item['name']} · {item['category']} · {money(item['amount_minor'])} · day {item['due_day']} · {'active' if item['active'] else 'paused'}")
            if c2.button("Pause" if item["active"] else "Activate",key=f"obl_{item['id']}"):
                set_recurring_active(uid,item["id"],not bool(item["active"])); st.rerun()
    st.caption("Aureli stores local account data in SQLite. Set AURELI_DB_PATH to a persistent mounted path in a hosted deployment. Never place this database in a public repository.")
