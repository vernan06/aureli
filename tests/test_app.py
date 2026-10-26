"""Streamlit page rendering and signup smoke test (run in .venv)."""
import os
import unittest
from datetime import date
from pathlib import Path
from uuid import uuid4

PROJECT=Path(__file__).resolve().parents[1]
TEMP_ROOT=PROJECT/".tmp"/"app-tests"
TEMP_ROOT.mkdir(parents=True,exist_ok=True)
os.environ["TEMP"]=str(TEMP_ROOT)
os.environ["TMP"]=str(TEMP_ROOT)
from streamlit.testing.v1 import AppTest

class StreamlitJourneyTests(unittest.TestCase):
    def test_signup_and_all_primary_sections_render(self):
        os.environ["AURELI_DB_PATH"]=str(TEMP_ROOT/f"aureli-{uuid4().hex}.sqlite3")
        app=AppTest.from_file(str(PROJECT/"streamlit_app.py"),default_timeout=30).run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        app.text_input[2].input("Aureli Demo").run()
        app.text_input[3].input("demo@example.test").run()
        app.text_input[4].input("a sufficiently long password").run()
        app.button[1].click().run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        app.radio[0].set_value("Financial Planner").run()
        next(w for w in app.text_input if w.label=="Category").input("Food").run()
        next(w for w in app.text_input if w.label=="Description").input("Groceries").run()
        next(w for w in app.number_input if w.label=="Amount (₹)").set_value(1250).run()
        next(w for w in app.button if w.label=="Add transaction").click().run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any(e.value=="Recent activity" for e in app.subheader))
        for section in ["Overview","EMI Prediction","Model Laboratory","Loan Simulator","Financial Planner"]:
            app.radio[0].set_value(section).run()
            self.assertEqual(len(app.exception),0,section)
        for subpage in ["Transactions","Budgets","Savings Goals","Financial Twin","Reports","Settings"]:
            app.radio[0].set_value("Financial Planner").run()
            app.radio[1].set_value(subpage).run()
            self.assertEqual(len(app.exception),0,subpage)

    def test_contributed_savings_sync_and_guided_loan_payment(self):
        os.environ["AURELI_DB_PATH"]=str(TEMP_ROOT/f"aureli-{uuid4().hex}.sqlite3")
        from aureli.auth.service import register
        from aureli.finance.service import add_goal,contribute_to_goal,goals_with_progress,set_budget,add_transaction
        from aureli.loans.service import save_loan,loan_balance,loan_payment_history
        from aureli.persistence.db import initialize
        initialize()
        user_id=register("journey@example.test","a sufficiently long password","Journey")
        goal_id=add_goal(user_id,"Emergency Fund",60000,5000)
        contribute_to_goal(user_id,goal_id,10000,date.today().isoformat(),"Seeded acceptance contribution")
        add_transaction(user_id,"expense",1000,"Food","Groceries",date.today().isoformat())
        set_budget(user_id,"Food",5000)
        loan_id=save_loan(user_id,"Reference loan",100000,12,12,date.today().isoformat())
        self.assertEqual(goals_with_progress(user_id)[0]["total_saved_minor"],1500000)

        app=AppTest.from_file(str(PROJECT/"streamlit_app.py"),default_timeout=30).run()
        app.session_state["user"]={"id":user_id,"name":"Journey"}
        app.run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any("₹15,000.00" in item.value for item in app.markdown))
        app.radio[0].set_value("Financial Planner").run()
        app.radio[1].set_value("Savings Goals").run()
        self.assertTrue(any("₹15,000.00" in item.value for item in app.markdown))
        app.radio[1].set_value("Budgets").run()
        self.assertTrue(any("₹4,000.00 remaining" in item.value for item in app.caption))
        app.radio[0].set_value("Loan Simulator").run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        total=next(item for item in app.number_input if item.label=="Total paid (₹)")
        principal=next(item for item in app.number_input if item.label=="Principal component (₹)")
        self.assertAlmostEqual(total.value,8884.88,places=2)
        self.assertAlmostEqual(principal.value,7884.88,places=2)
        next(item for item in app.button if item.label=="Record payment").click().run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertEqual(loan_balance(user_id,loan_id),9211512)
        self.assertEqual(len(loan_payment_history(user_id,loan_id)),1)
        self.assertTrue(any(item.label=="Baseline loan" for item in app.selectbox))
        self.assertTrue(any(item.label=="Interest saved" for item in app.metric))

    @unittest.skipUnless((PROJECT/"artifacts/models/emi_eligibility.joblib").is_file() and (PROJECT/"artifacts/models/max_monthly_emi.joblib").is_file(),"authorized local model artifacts are not present")
    def test_real_saved_model_inference_and_model_laboratory(self):
        os.environ["AURELI_DB_PATH"]=str(TEMP_ROOT/f"aureli-{uuid4().hex}.sqlite3")
        from aureli.auth.service import register
        from aureli.ml.inference import load_artifact,predict
        from aureli.persistence.db import initialize
        initialize(); user_id=register("model@example.test","a sufficiently long password","Model")
        artifact=load_artifact("emi_eligibility")
        malformed={name:None for name in artifact["input_feature_columns"]}; malformed["monthly_salary"]="not a number"
        with self.assertRaisesRegex(ValueError,"must be numeric"):
            predict("emi_eligibility",malformed)

        app=AppTest.from_file(str(PROJECT/"streamlit_app.py"),default_timeout=90).run()
        app.session_state["user"]={"id":user_id,"name":"Model"}; app.run()
        app.radio[0].set_value("EMI Prediction").run()
        salaries=[item for item in app.text_input if item.label=="Monthly salary (₹)"]
        self.assertEqual(len(salaries),2)
        for item in salaries: item.input("55000")
        app.run()
        prediction_buttons=[item for item in app.button if item.label=="Predict with saved Aureli model"]
        prediction_buttons[0].click().run()
        self.assertTrue(any(item.label=="Model assessment" for item in app.metric))
        self.assertEqual(len(app.exception),0,str(app.exception))
        prediction_buttons=[item for item in app.button if item.label=="Predict with saved Aureli model"]
        prediction_buttons[1].click().run()
        self.assertTrue(any(item.label=="Estimated maximum monthly EMI" for item in app.metric))
        self.assertEqual(len(app.exception),0,str(app.exception))
        app.radio[0].set_value("Model Laboratory").run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any(item.value=="Six-model comparison" for item in app.subheader))

if __name__=="__main__": unittest.main()
