"""Streamlit page rendering and signup smoke test (run in .venv)."""
import os
import unittest
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

if __name__=="__main__": unittest.main()
