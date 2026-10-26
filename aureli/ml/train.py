"""CLI entrypoint for the optional offline model training workflow."""
import argparse
import json
from aureli.ml.training import train_all

def main():
    parser=argparse.ArgumentParser(description="Train Aureli's six EMI models on an explicitly supplied, rights-cleared CSV.")
    parser.add_argument("dataset_csv",help="Path to a dataset with emi_eligibility and max_monthly_emi targets")
    parser.add_argument("--tracking-uri",default="file:./mlruns",help="Local MLflow tracking URI (default: file:./mlruns)")
    parser.add_argument("--confirm-data-rights",action="store_true",help="Affirm that you are authorized to train on this dataset")
    parser.add_argument("--quick",action="store_true",help="Use one deterministic parameter candidate per model for pipeline smoke tests")
    args=parser.parse_args()
    print(json.dumps(train_all(args.dataset_csv,args.tracking_uri,data_authorized=args.confirm_data_rights,quick=args.quick),indent=2))

if __name__=="__main__": main()
