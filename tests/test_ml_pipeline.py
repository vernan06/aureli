"""The synthetic fixture exercises pipeline mechanics only, never product metrics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from aureli.ml.inference import load_artifact, predict
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
            for target in ("emi_eligibility","max_monthly_emi"):
                artifact=load_artifact(target,artifact_dir)
                self.assertEqual(len(artifact["dataset_sha256"]),64)
                values={"monthly_salary":"80000","current_emi_amount":"12000","monthly_rent":"15000","employment_type":"Salaried"}
                result=predict(target,values,artifact_dir)
                self.assertEqual(result["target"],target)
                self.assertIn("prediction",result)
            client=MlflowClient(tracking_uri=tracking)
            experiments=client.search_experiments()
            named={e.name:e for e in experiments}
            self.assertTrue({"aureli-classification-emi_eligibility","aureli-regression-max_monthly_emi"}.issubset(named))
            recorded_run_ids={run.info.run_id for name in ("aureli-classification-emi_eligibility","aureli-regression-max_monthly_emi") for run in client.search_runs([named[name].experiment_id])}
            self.assertTrue({item["run_id"] for item in models}.issubset(recorded_run_ids))

if __name__=="__main__": unittest.main()
