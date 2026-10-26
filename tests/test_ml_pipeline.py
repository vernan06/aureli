"""The synthetic fixture exercises pipeline mechanics only, never product metrics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from aureli.ml.inference import load_artifact, predict
from aureli.ml.pipeline import _clean_frame,_deps,_engineer,_metrics
from aureli.ml.training import inspect_dataset, train_all, validate_dataset

PROJECT=Path(__file__).resolve().parents[1]
TEMP_ROOT=PROJECT/".tmp"
TEMP_ROOT.mkdir(parents=True,exist_ok=True)

class MLPipelineTests(unittest.TestCase):
    def test_schema_validation_and_profile(self):
        with self.assertRaises(ValueError): validate_dataset(pd.DataFrame({"feature":[1]}))
        frame=pd.DataFrame({"salary":[10,20,30,40],"emi_eligibility":["A","B","A","B"],"max_monthly_emi":[1,2,3,4]})
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            path=Path(temp)/"data.csv"; frame.to_csv(path,index=False)
            profile=inspect_dataset(path)
            self.assertEqual(profile["rows"],4)
            self.assertEqual(profile["eligibility_distribution"],{"A":2,"B":2})

    def test_unambiguous_serialization_and_gender_normalization(self):
        d=_deps(); frame=pd.DataFrame({"monthly_salary":["18000.0.0","20000.0"]+[f"{30000+i}.0" for i in range(10)],
            "bank_balance":["nan.0","12000.0.0"]+[f"{10000+i}.0" for i in range(10)],
            "gender":["MALE","f"]+["male","female"]*5,
            "employment_type":["Private","Government"]+["Self-employed"]*9+[None]})
        cleaned,repairs,malformed,numeric_repairs=_clean_frame(frame,d["pd"])
        self.assertEqual(cleaned["monthly_salary"].iloc[0],18000)
        self.assertEqual(cleaned["monthly_salary"].iloc[1],20000)
        self.assertTrue(pd.isna(cleaned["bank_balance"].iloc[0]))
        self.assertEqual(cleaned["gender"].iloc[:2].tolist(),["Male","Female"])
        self.assertGreater(numeric_repairs["monthly_salary"],0)
        self.assertNotIn("monthly_salary",malformed)
        engineered=_engineer(frame,d["pd"])
        self.assertEqual(engineered["gender"].dtype,object)
        self.assertTrue(np.isnan(engineered["employment_type"].iloc[-1]))

    def test_classification_metrics_include_calibration_diagnostics(self):
        d=_deps(); y=np.array([0,1,2,0,1,2]); probabilities=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.2,.7],[.6,.3,.1],[.2,.6,.2],[.2,.2,.6]])
        result=_metrics("classification",y,np.array([0,1,2,0,1,2]),d,["A","B","C"],probabilities)
        self.assertEqual(result["per_class"]["A"]["recall"],1.0)
        self.assertIn("multiclass_brier_score",result)
        self.assertIn("expected_calibration_error_10_bins",result)
        self.assertTrue(result["calibration_bins"])

    def test_six_model_mlflow_artifacts_and_inference(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            root=Path(temp); rng=np.random.default_rng(42); rows=120
            salary=rng.integers(25000,150000,rows).astype(float)
            current=rng.integers(0,30000,rows).astype(float)
            frame=pd.DataFrame({
                "monthly_salary":salary,
                "current_emi_amount":current,
                "monthly_rent":rng.integers(0,45000,rows),
                "employment_type":np.resize(["Salaried","Self-employed","Contract"],rows),
                "emi_eligibility":np.resize(["Low_Risk","Medium_Risk","High_Risk"],rows),
                "max_monthly_emi":np.maximum(500,salary*.35-current).round(2),
            })
            frame["monthly_salary"]=frame["monthly_salary"].astype(object)
            frame.loc[0,"monthly_salary"]="malformed-but-repairable"
            frame.loc[1,"monthly_rent"]=np.nan
            frame=pd.concat([frame,frame.iloc[[5]]],ignore_index=True)
            csv_path=root/"cleared-test-fixture.csv"; frame.to_csv(csv_path,index=False)
            tracking=(root/"mlruns").as_uri(); artifact_dir=root/"models"
            with self.assertRaises(PermissionError): train_all(csv_path,tracking_uri=tracking,artifact_dir=artifact_dir)
            results=train_all(csv_path,tracking_uri=tracking,artifact_dir=artifact_dir,data_authorized=True,quick=True)
            models=[item for item in results if item.get("model") in {"logistic_regression","linear_regression","random_forest","xgboost"} and not item.get("registered_model")]
            self.assertEqual(len(models),6)
            self.assertTrue((artifact_dir/"emi_eligibility.joblib").is_file())
            self.assertTrue((artifact_dir/"max_monthly_emi.joblib").is_file())
            self.assertTrue((artifact_dir/"model_comparison.csv").is_file())
            self.assertTrue((artifact_dir/"data_quality_report.md").is_file())
            comparison=pd.read_csv(artifact_dir/"model_comparison.csv")
            self.assertEqual(len(comparison),6)
            for target in ("emi_eligibility","max_monthly_emi"):
                artifact=load_artifact(target,artifact_dir)
                self.assertEqual(len(artifact["dataset_sha256"]),64)
                values={"monthly_salary":"80000","current_emi_amount":"12000","monthly_rent":"15000","employment_type":"Salaried"}
                result=predict(target,values,artifact_dir)
                self.assertEqual(result["target"],target)
                self.assertIn("prediction",result)
                if target=="max_monthly_emi": self.assertGreaterEqual(result["prediction"],artifact["target_min"])
            client=MlflowClient(tracking_uri=tracking)
            experiments=client.search_experiments()
            named={e.name:e for e in experiments}
            self.assertTrue({"aureli-classification-emi_eligibility","aureli-regression-max_monthly_emi"}.issubset(named))
            recorded_run_ids={run.info.run_id for name in ("aureli-classification-emi_eligibility","aureli-regression-max_monthly_emi") for run in client.search_runs([named[name].experiment_id])}
            self.assertTrue({item["run_id"] for item in models}.issubset(recorded_run_ids))

if __name__=="__main__": unittest.main()
