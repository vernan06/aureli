import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from aureli.auth.service import authenticate, delete_local_account, register, verify_password
from aureli.finance.service import (add_category, add_goal, add_recurring_obligation, add_transaction,
    budgets, categories, contribute_to_goal, delete_transaction, goal_contributions, goals_with_progress,
    monthly_history, monthly_summary, recurring_obligations, set_budget, set_recurring_active,
    transactions, update_transaction)
from aureli.loans.service import (amortization_schedule, compare_debt_payoff, loan_balance,
    loan_payment_history, loan_summary, record_loan_payment, save_loan, user_loans)
from aureli.intelligence.service import recommendations
from aureli.intelligence.scenario import forecast_from_history, simulate_monthly_plan
from aureli.persistence.db import connect, initialize


class AureliCoreTests(unittest.TestCase):
    def setUp(self):
        test_root=Path(__file__).resolve().parents[1]/".tmp"/"tests"
        test_root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=test_root)
        self.env=patch.dict(os.environ,{"AURELI_DB_PATH":str(Path(self.tmp.name)/"db.sqlite3")})
        self.env.start(); initialize()
        self.user=register("one@example.test","correct horse battery", "One")
        self.other=register("two@example.test","another safe password", "Two")

    def tearDown(self):
        self.env.stop(); self.tmp.cleanup()

    def test_password_hash_and_authentication(self):
        self.assertIsNotNone(authenticate("ONE@example.test","correct horse battery"))
        self.assertIsNone(authenticate("one@example.test","wrong password"))
        with connect() as db: encoded=db.execute("SELECT password_hash FROM users WHERE id=?",(self.user,)).fetchone()[0]
        self.assertTrue(verify_password("correct horse battery",encoded))
        self.assertFalse(verify_password("wrong password",encoded))

    def test_guest_account_deletion_cascades_only_for_selected_owner(self):
        transaction=add_transaction(self.user,"expense",10,"Food","guest test","2026-10-01")
        self.assertTrue(delete_local_account(self.user))
        self.assertFalse(delete_local_account(self.user))
        self.assertEqual(len(transactions(self.other)),0)
        with connect() as db:
            self.assertIsNone(db.execute("SELECT id FROM transactions WHERE id=?",(transaction,)).fetchone())
            self.assertIsNotNone(db.execute("SELECT id FROM users WHERE id=?",(self.other,)).fetchone())

    def test_transaction_owner_boundary_and_month_summary(self):
        txid=add_transaction(self.user,"income","1234.565","Salary","pay","2026-10-01")
        add_transaction(self.user,"expense","34.50","Food","market","2026-10-02")
        add_transaction(self.other,"income","9000","Salary","private","2026-10-02")
        self.assertEqual(monthly_summary(self.user,"2026-10"),{"income":123457,"expenses":3450,"surplus":120007})
        self.assertEqual(len(transactions(self.user)),2)
        self.assertFalse(delete_transaction(self.other,txid))
        self.assertTrue(update_transaction(self.user,txid,kind="income",amount=1300,category="Salary",description="updated",occurred_on="2026-10-03"))
        self.assertTrue(delete_transaction(self.user,txid))

    def test_money_validation_and_budget_spend(self):
        with self.assertRaises(ValueError): add_transaction(self.user,"expense",0,"Food","","2026-10-01")
        for invalid in ("nan","inf","not money",-1):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError): add_transaction(self.user,"expense",invalid,"Food","","2026-10-01")
        add_transaction(self.user,"expense",25,"Food","","2026-10-03")
        set_budget(self.user,"Food",100)
        rows=budgets(self.user)
        self.assertEqual(rows[0]["spent_minor"],2500)
        self.assertEqual(rows[0]["monthly_limit_minor"],10000)

    def test_category_migration_and_recurring_obligations_are_user_scoped(self):
        add_category(self.user,"Rent","expense")
        add_category(self.other,"Salary","income")
        self.assertEqual([(r["name"],r["kind"]) for r in categories(self.user)],[ ("Rent","expense") ])
        obligation=add_recurring_obligation(self.user,"House rent","Rent",25000,5)
        self.assertEqual(len(recurring_obligations(self.user)),1)
        self.assertEqual(len(recurring_obligations(self.other)),0)
        self.assertTrue(set_recurring_active(self.user,obligation,False))
        self.assertEqual(len(recurring_obligations(self.user)),0)
        self.assertEqual(len(recurring_obligations(self.user,active_only=False)),1)

    def test_goal_contributions_update_progress_and_owner_boundary(self):
        goal=add_goal(self.user,"Trip",1000,100)
        contribute_to_goal(self.user,goal,200,"2026-10-10","Monthly saving")
        self.assertEqual(goals_with_progress(self.user)[0]["total_saved_minor"],30000)
        self.assertEqual(len(goal_contributions(self.user,goal)),1)
        with self.assertRaises(ValueError): contribute_to_goal(self.other,goal,100,"2026-10-10")
        with self.assertRaises(ValueError): contribute_to_goal(self.user,goal,800,"2026-10-10")

    def test_loan_payment_history_reduces_principal_and_is_owner_scoped(self):
        loan=save_loan(self.user,"Auto",120000,12,12,"2026-01-01")
        record_loan_payment(self.user,loan,10000,9000,"2026-02-01","statement")
        record_loan_payment(self.user,loan,5000,5000,"2026-02-15","prepayment","extra_principal")
        self.assertEqual(loan_balance(self.user,loan),10600000)
        self.assertEqual(len(loan_payment_history(self.user,loan)),2)
        self.assertEqual(loan_payment_history(self.other,loan),[])
        with self.assertRaises(ValueError): record_loan_payment(self.other,loan,1000,900,"2026-03-01")

    def test_loan_payment_rejects_placeholder_and_duplicate_submissions(self):
        loan=save_loan(self.user,"Home",100000,12,12,"2026-01-01")
        with self.assertRaisesRegex(ValueError,"below ₹1.00"):
            record_loan_payment(self.user,loan,.01,0,"2026-02-01","First EMI test")
        with self.assertRaisesRegex(ValueError,"Zero-principal"):
            record_loan_payment(self.user,loan,8884.88,0,"2026-02-01","First EMI test")
        record_loan_payment(self.user,loan,8884.88,7884.88,"2026-02-01","First EMI test")
        with self.assertRaisesRegex(ValueError,"already recorded"):
            record_loan_payment(self.user,loan,8884.88,7884.88,"2026-02-01","First EMI test")
        self.assertEqual(len(loan_payment_history(self.user,loan)),1)

    def test_reference_loan_and_accelerated_payoff_rounding(self):
        regular=loan_summary(100000,12,12)
        faster=loan_summary(100000,12,12,2000)
        self.assertEqual(regular["emi"],Decimal("8884.88"))
        self.assertEqual(regular["months"],12)
        self.assertEqual(faster["months"],10)
        self.assertEqual(faster["total_interest"],Decimal("5431.07"))
        self.assertEqual(loan_summary(1000,0,3)["schedule"][-1]["balance"],0)

    def test_loan_inputs_reject_nonfinite_negative_and_fractional_terms(self):
        for rate,months in ((-1,12),("nan",12),(12,0),(12,1.5),(12,601)):
            with self.subTest(rate=rate,months=months),self.assertRaises(ValueError): loan_summary(100000,rate,months)
        for rate,months in ((-1,12),("inf",12),(12,1.2),(12,601)):
            with self.subTest(rate=rate,months=months),self.assertRaises(ValueError): save_loan(self.user,"Invalid loan",1000,rate,months,"2026-10-01")

    def test_snowball_avalanche_and_what_if_are_non_mutating(self):
        small=save_loan(self.user,"Small high rate",10000,18,24,"2026-01-01")
        large=save_loan(self.user,"Large low rate",50000,5,48,"2026-01-01")
        before=len(user_loans(self.user))
        records=user_loans(self.user)
        snowball=compare_debt_payoff(records,500,"snowball")
        avalanche=compare_debt_payoff(records,500,"avalanche")
        self.assertTrue(snowball["paid_off"] and avalanche["paid_off"])
        self.assertEqual(list(snowball["payoff_months"])[0],"Small high rate")
        self.assertEqual(len(user_loans(self.user)),before)
        plan=simulate_monthly_plan(income=80000,expenses=45000,existing_emi=9000,income_change_pct=10,
            expense_reduction_pct=10,new_loan_principal=100000,new_loan_rate=12,new_loan_months=12,
            extra_debt_payment=1000,monthly_savings=5000,initial_savings=20000,projection_months=12)
        self.assertGreater(plan["projected_savings"],Decimal("20000"))
        self.assertGreater(plan["new_loan_emi"],0)
        self.assertEqual(len(user_loans(self.user)),before)

    def test_cash_flow_history_and_projection_use_observed_months(self):
        add_transaction(self.user,"income",1000,"Salary","","2026-08-01")
        add_transaction(self.user,"expense",700,"Rent","","2026-08-03")
        add_transaction(self.user,"income",1200,"Salary","","2026-09-01")
        add_transaction(self.user,"expense",800,"Rent","","2026-09-03")
        forecast=forecast_from_history(monthly_history(self.user),500,6)
        self.assertEqual(forecast["months"],2)
        self.assertEqual(forecast["average_income"],110000)
        self.assertEqual(forecast["average_expenses"],75000)

    def test_goal_validation(self):
        self.assertEqual(add_goal(self.user,"Reserve",1000,250),1)
        with self.assertRaises(ValueError): add_goal(self.user,"Invalid",100,101)

    def test_recommendations_use_stored_data(self):
        add_transaction(self.user,"income",1000,"Salary","","2026-10-01")
        add_transaction(self.user,"expense",900,"Rent","","2026-10-02")
        items=recommendations(self.user,"2026-10")
        self.assertTrue(any("cash reserve" in row["title"].lower() for row in items))
        self.assertFalse(any("credit score" in row["title"].lower() for row in items))

    def test_zero_interest_and_final_reconciliation(self):
        rows=amortization_schedule("1000",0,3)
        self.assertEqual([r["payment"] for r in rows],[__import__("decimal").Decimal("333.33"),__import__("decimal").Decimal("333.33"),__import__("decimal").Decimal("333.34")])
        self.assertEqual(rows[-1]["balance"],0)

    def test_interest_schedule_and_extra_payment(self):
        normal=loan_summary(10000,12,12)
        accelerated=loan_summary(10000,12,12,100)
        self.assertEqual(normal["schedule"][-1]["balance"],0)
        self.assertLess(accelerated["months"],normal["months"])
        self.assertLess(accelerated["total_interest"],normal["total_interest"])


if __name__=="__main__": unittest.main()
