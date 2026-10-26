"""CLI entrypoint for the optional offline model training workflow."""
import argparse
import json
from pathlib import Path
from aureli.ml.training import inspect_dataset,train_all

def main():
    parser=argparse.ArgumentParser(description="Train Aureli's six EMI models on an explicitly supplied, rights-cleared CSV.")
    parser.add_argument("dataset_csv",help="Path to a dataset with emi_eligibility and max_monthly_emi targets")
    parser.add_argument("--tracking-uri",default="file:./mlruns",help="Local MLflow tracking URI (default: file:./mlruns)")
    parser.add_argument("--artifact-dir",default="artifacts/models",help="Where to save local model and report artifacts")
    parser.add_argument("--max-rows",type=int,default=160000,help="Reproducible representative sample cap for memory-bounded training")
    parser.add_argument("--confirm-data-rights",action="store_true",help="Affirm that you are authorized to train on this dataset")
    parser.add_argument("--quick",action="store_true",help="Use one deterministic parameter candidate per model for pipeline smoke tests")
    parser.add_argument("--inspect-only",action="store_true",help="Write a dataset profile and quality report without training")
    args=parser.parse_args()
    if args.inspect_only:
        from aureli.ml.pipeline import _write_quality_report
        profile=inspect_dataset(args.dataset_csv); output=Path(args.artifact_dir); output.mkdir(parents=True,exist_ok=True)
        prior=output/"dataset_profile.json"
        if prior.is_file():
            try:
                previous=json.loads(prior.read_text(encoding="utf-8"))
                if previous.get("sha256")==profile.get("sha256"):
                    for key in ("rows_retained_after_cleaning","training_rows_used","valid_rows_not_sampled","sampling_method","split_plan","cleaning_decisions","excluded_records"):
                        if key in previous: profile[key]=previous[key]
            except (OSError,ValueError): pass
        (output/"dataset_profile.json").write_text(json.dumps(profile,indent=2,default=str),encoding="utf-8")
        _write_quality_report(profile,output/"data_quality_report.md")
        print(json.dumps({"dataset_profile":str(output/"dataset_profile.json"),"quality_report":str(output/"data_quality_report.md")},indent=2))
    else:
        print(json.dumps(train_all(args.dataset_csv,args.tracking_uri,args.artifact_dir,
            data_authorized=args.confirm_data_rights,quick=args.quick,max_rows=args.max_rows),indent=2))

if __name__=="__main__": main()
