"""Reproducible EMI classification/regression with MLflow and local artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

TARGETS={"classification":"emi_eligibility","regression":"max_monthly_emi"}
RANDOM_STATE=42
MAX_TRAIN_ROWS=120_000

def _deps():
    import joblib
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression,LogisticRegression
    from sklearn.metrics import (accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,log_loss,
        f1_score,mean_absolute_error,mean_squared_error,precision_score,recall_score,r2_score,roc_auc_score)
    from sklearn.model_selection import (GridSearchCV,RandomizedSearchCV,StratifiedKFold,train_test_split)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder,RobustScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier,XGBRegressor
    dependencies=locals()
    dependencies["mlflow_sklearn"]=mlflow.sklearn
    dependencies["log_loss"]=log_loss
    return dependencies

def validate_dataset(frame):
    required=set(TARGETS.values())
    missing=required-set(frame.columns)
    if missing: raise ValueError("Missing required target column(s): "+", ".join(sorted(missing)))
    if len(frame)==0: raise ValueError("Dataset contains no rows.")
    feature_columns=[c for c in frame.columns if c not in required and c.lower() not in {"id","loan_id","customer_id","user_id","application_id"}]
    if not feature_columns: raise ValueError("Dataset must contain at least one input feature.")
    return feature_columns

def _clean_frame(frame,pd):
    data=frame.copy()
    data.columns=[str(c).strip() for c in data.columns]
    repairs={}; malformed={}; numeric_repairs={}
    for col in data.select_dtypes(include=["object","string"]).columns:
        original=data[col].astype("string")
        cleaned=original.str.strip()
        cleaned=cleaned.mask(cleaned.str.casefold().isin(["","nan","none","null","na"]),pd.NA)
        if col.lower()=="gender":
            canonical=cleaned.str.casefold().map({"m":"Male","male":"Male","f":"Female","female":"Female"})
            cleaned=canonical.fillna(cleaned)
        if col in TARGETS.values():
            repairs[col]=int((cleaned.fillna("<NA>")!=original.fillna("<NA>")).sum())
            data[col]=cleaned
            continue
        nonempty=int(cleaned.notna().sum())
        sample=cleaned.dropna()
        sample=sample.sample(n=min(2048,len(sample)),random_state=RANDOM_STATE) if len(sample)>2048 else sample
        sample_numeric=pd.to_numeric(sample,errors="coerce")
        numeric_like=bool(nonempty and len(sample) and sample_numeric.notna().sum()>=math.ceil(.8*len(sample)))
        if numeric_like:
            # Repair only numeric-looking columns. Ordinary decimal strings stay
            # intact, while duplicated serialization suffixes and nan.0 are fixed.
            before_numeric_repair=cleaned.copy()
            repeated=cleaned.str.contains(r"\.0\.0$",regex=True,na=False)
            if repeated.any():
                values=cleaned.loc[repeated]
                for _ in range(3): values=values.str.replace(r"^([+-]?\d+(?:\.0)*)\.0\.0$",r"\1.0",regex=True)
                cleaned.loc[repeated]=values
            cleaned=cleaned.str.replace(r"^nan\.0$","nan",case=False,regex=True)
            cleaned=cleaned.mask(cleaned.str.casefold().isin(["nan","none","null","na"]),pd.NA)
            numeric_repairs[col]=int((cleaned.fillna("<NA>")!=before_numeric_repair.fillna("<NA>")).sum())
            repairs[col]=int((cleaned.fillna("<NA>")!=original.fillna("<NA>")).sum())
            numeric=pd.to_numeric(cleaned,errors="coerce")
            malformed_count=int((cleaned.notna()&numeric.isna()).sum())
            if malformed_count: malformed[col]=malformed_count
            data[col]=numeric
        else:
            repairs[col]=int((cleaned.fillna("<NA>")!=original.fillna("<NA>")).sum())
            numeric_repairs[col]=0
            data[col]=cleaned
    if TARGETS["regression"] in data:
        data[TARGETS["regression"]]=pd.to_numeric(data[TARGETS["regression"]],errors="coerce")
    return data,repairs,malformed,numeric_repairs

def _engineer(frame, pd):
    data,_,_,_=_clean_frame(frame,pd)
    feature_names=set(data.columns)-set(TARGETS.values())
    if {"monthly_salary","current_emi_amount"}<=feature_names:
        salary=pd.to_numeric(data["monthly_salary"],errors="coerce")
        emi=pd.to_numeric(data["current_emi_amount"],errors="coerce")
        data["current_emi_to_salary_ratio"]=emi/salary.where(salary>0)
        data["salary_missing_flag"]=salary.isna().astype(int)
    expense_cols=[c for c in ("monthly_rent","school_fees","college_fees","travel_expenses","groceries_utilities","other_monthly_expenses") if c in feature_names]
    if expense_cols and "monthly_salary" in feature_names:
        numeric_exp=data[expense_cols].apply(pd.to_numeric,errors="coerce")
        data["estimated_monthly_expenses"]=numeric_exp.sum(axis=1,min_count=1)
        salary=pd.to_numeric(data["monthly_salary"],errors="coerce")
        data["estimated_expense_ratio"]=data["estimated_monthly_expenses"]/salary.where(salary>0)
    for col in data.select_dtypes(include=["number"]).columns:
        data[col]=data[col].replace([float("inf"),float("-inf")],float("nan"))
    for col in data.select_dtypes(include=["object","string"]).columns:
        data[col]=data[col].astype(object).where(data[col].notna(),float("nan"))
    return data

def inspect_dataset(csv_path):
    d=_deps(); pd=d["pd"]; path=Path(csv_path); frame=pd.read_csv(path,low_memory=False)
    features=validate_dataset(frame); cleaned,repairs,malformed,numeric_repairs=_clean_frame(frame,pd)
    numeric=cleaned[features].select_dtypes(include=["number","bool"])
    outliers={}
    for col in numeric:
        series=numeric[col].dropna()
        if series.empty: continue
        q1,q3=series.quantile(.25),series.quantile(.75); spread=q3-q1
        count=int(((series<q1-1.5*spread)|(series>q3+1.5*spread)).sum()) if spread else 0
        if count: outliers[col]=count
    eligibility=cleaned[TARGETS["classification"]].dropna().astype(str).value_counts().to_dict()
    target_summary={}
    for target in TARGETS.values():
        values=cleaned[target].dropna()
        if target==TARGETS["classification"]: target_summary[target]={"counts":{str(k):int(v) for k,v in values.value_counts().items()}}
        else: target_summary[target]={"count":int(values.size),"min":float(values.min()),"max":float(values.max()),"mean":float(values.mean()),"quantiles":{str(q):float(values.quantile(q)) for q in [0,.01,.1,.25,.5,.75,.9,.99,1]}}
    feature_distributions={}
    for col in features:
        series=cleaned[col]
        if col in numeric:
            s=series.dropna()
            feature_distributions[col]={"kind":"numeric","count":int(s.size),"missing":int(series.isna().sum()),"mean":float(s.mean()) if len(s) else None,"std":float(s.std()) if len(s)>1 else None,"min":float(s.min()) if len(s) else None,"quantiles":{str(q):float(s.quantile(q)) for q in [0,.01,.25,.5,.75,.99,1]} if len(s) else {},"max":float(s.max()) if len(s) else None}
        else:
            counts=series.fillna("<MISSING>").astype(str).value_counts(dropna=False)
            feature_distributions[col]={"kind":"categorical","count":int(series.notna().sum()),"missing":int(series.isna().sum()),"unique":int(series.nunique(dropna=True)),"counts":{str(k):int(v) for k,v in counts.items()}}
    correlation={}
    for target in (TARGETS["regression"],):
        y=cleaned[target]
        for col in numeric.columns:
            if col==target or cleaned[col].nunique(dropna=True)<2: continue
            corr=cleaned[col].corr(y)
            if pd.notna(corr): correlation[col]=float(corr)
    exact_target_copies={target:[col for col in features if cleaned[col].reset_index(drop=True).equals(cleaned[target].reset_index(drop=True))] for target in TARGETS.values()}
    class_feature_means={str(label):{str(col):float(value) for col,value in values.items() if pd.notna(value)}
        for label,values in cleaned.groupby(TARGETS["classification"])[numeric.columns].mean().to_dict(orient="index").items()}
    return {"dataset_file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows":int(len(frame)),"rows_original":int(len(frame)),"columns":list(frame.columns),"feature_count":len(features),
        "raw_exact_duplicates":int(frame.duplicated().sum()),"duplicates_after_cleaning":int(cleaned.duplicated().sum()),
        "missing_raw":{str(k):int(v) for k,v in frame.isna().sum().items() if v},
        "missing_after_cleaning":{str(k):int(v) for k,v in cleaned.isna().sum().items() if v},
        "string_format_repairs":{k:v for k,v in repairs.items() if v},"numeric_serialization_repairs":{k:v for k,v in numeric_repairs.items() if v},"unparsed_numeric_text":malformed,
        "outlier_counts_iqr_not_removed":outliers,"target_distributions":target_summary,
        "eligibility_distribution":eligibility,
        "class_imbalance_ratio_largest_to_smallest":float(max(eligibility.values())/min(eligibility.values())) if eligibility else None,
        "feature_distributions":feature_distributions,"regression_target_feature_correlations_pearson":dict(sorted(correlation.items(),key=lambda kv:abs(kv[1]),reverse=True)),
        "classification_numeric_feature_means_by_label":class_feature_means,
        "target_leakage_review":{"target_columns_excluded_from_features":True,"exact_target_copy_features":exact_target_copies,
            "note":"No direct copies of the target columns are used as features. Several financial/request inputs may be used when creating the labels; the company label-generation rules were not supplied, so target provenance should be confirmed and unusually strong test scores reviewed."}}

def _write_quality_report(profile,path):
    lines=["# EMI dataset quality and leakage report","",
        f"- Source file: `{profile['dataset_file']}` (original kept unchanged)",
        f"- SHA-256: `{profile['sha256']}`",
        f"- Original rows: {profile['rows_original']:,}",
        f"- Rows retained after cleaning: {profile.get('rows_retained_after_cleaning',profile['rows_original']):,}",
        f"- Rows used for training: {profile.get('training_rows_used',profile['rows_original']):,}",
        f"- Valid rows not sampled: {profile.get('valid_rows_not_sampled',0):,}",
        f"- Original features: {profile['feature_count']}","",
        "## Excluded records","",
        "| Reason | Rows |","|---|---:|"]
    lines.extend(f"| {reason.replace('_',' ')} | {count:,} |" for reason,count in profile.get("excluded_records",{}).items())
    lines.extend(["","## Missing and malformed values","", "Raw missing values by column:"])
    lines.extend([f"- {name}: {count:,}" for name,count in profile.get("missing_raw",{}).items()] or ["- None detected."])
    lines.extend(["","Clearly repairable string formatting variants normalized (counts by column):"])
    lines.extend([f"- {name}: {count:,}" for name,count in profile.get("string_format_repairs",{}).items()] or ["- None detected."])
    lines.append("Numeric serialization suffixes repaired (counts by column):")
    lines.extend([f"- {name}: {count:,}" for name,count in profile.get("numeric_serialization_repairs",{}).items()] or ["- None detected."])
    lines.append("Unparsed numeric text retained as missing for train-fold imputation:")
    lines.extend([f"- {name}: {count:,}" for name,count in profile.get("unparsed_numeric_text",{}).items()] or ["- None detected."])
    lines.append("Remaining missing values after formatting cleanup:")
    lines.extend([f"- {name}: {count:,}" for name,count in profile.get("missing_after_cleaning",{}).items()] or ["- None detected."])
    lines.extend(["","## Target distributions",""])
    for target,summary in profile.get("target_distributions",{}).items():
        lines.extend([f"### `{target}`",""])
        if "counts" in summary:
            lines.extend(["| Label | Count |","|---|---:|"]+[f"| {label} | {count:,} |" for label,count in summary["counts"].items()])
        else:
            lines.append(f"Count {summary['count']:,}; min {summary['min']:,.2f}; mean {summary['mean']:,.2f}; max {summary['max']:,.2f}.")
            lines.append("Quantiles: "+", ".join(f"{q}: {v:,.2f}" for q,v in summary["quantiles"].items()))
        lines.append("")
    lines.extend([f"Class imbalance ratio (largest/smallest): {profile.get('class_imbalance_ratio_largest_to_smallest'):.2f}:1","",
        "## Feature distributions","", "Numeric features report count, missingness, mean, standard deviation, min/max and selected quantiles. Categorical features list every observed value count.",""])
    for name,dist in profile.get("feature_distributions",{}).items():
        lines.append(f"### `{name}`")
        if dist["kind"]=="numeric":
            lines.append(f"Count {dist['count']:,}; missing {dist['missing']:,}; mean {dist['mean']}; std {dist['std']}; min {dist['min']}; max {dist['max']}.")
            lines.append("Quantiles: "+", ".join(f"{q}: {v}" for q,v in dist["quantiles"].items()))
        else:
            lines.append(f"Count {dist['count']:,}; missing {dist['missing']:,}; distinct {dist['unique']}.")
            lines.extend(f"- {value}: {count:,}" for value,count in dist["counts"].items())
        lines.append("")
    lines.extend(["## Potential target leakage","",profile["target_leakage_review"]["note"],
        "Both target columns are explicitly removed from the feature matrix. No input columns named as either target are permitted. Correlations below are diagnostics, not proof of leakage.",""])
    for target,copies in profile["target_leakage_review"].get("exact_target_copy_features",{}).items():
        lines.append(f"- Exact feature copies of `{target}`: {', '.join(copies) if copies else 'none detected'}.")
    lines.append("")
    lines.extend(f"- Pearson correlation with `max_monthly_emi`: `{name}` = {value:.4f}" for name,value in profile.get("regression_target_feature_correlations_pearson",{}).items())
    lines.extend(["","Numeric feature means by eligibility class (descriptive relationship checks):",""])
    for label,values in profile.get("classification_numeric_feature_means_by_label",{}).items():
        lines.append(f"- {label}: "+", ".join(f"{name}={value:,.2f}" for name,value in values.items()))
    lines.extend(["","IQR outlier counts, diagnostic only:",""])
    lines.extend([f"- {name}: {count:,}" for name,count in profile.get("outlier_counts_iqr_not_removed",{}).items()] or ["- None detected."])
    lines.extend(["","## Cleaning decisions","",f"- Exact duplicates in the raw file: {profile.get('raw_exact_duplicates',0):,}; duplicates after deterministic cleanup: {profile.get('duplicates_after_cleaning',0):,}.",
        "- No financial feature values were fabricated. Missing/unparsed feature values remain missing until imputation is fit within each training fold.",
        "- Rows with missing/invalid target values are excluded; duplicate rows are excluded.",
        "- IQR outlier counts are diagnostic only; no outlier rows or target values were capped or removed.",
        "- Numeric text was normalized only when at least 80% of nonmissing values parse as numeric; ambiguous fields remain categorical.",
        "- Gender formatting variants (M/male/MALE and F/female/FEMALE) are canonicalized.",
        "- Training uses a reproducible proportional sample stratified by eligibility label and EMI target quantile when the valid set exceeds the local row cap.",
        "- Split is 60/20/20 train/validation/untouched test; two-fold CV selects hyperparameters; the test partition is not used for model selection.",""])
    Path(path).write_text("\n".join(lines),encoding="utf-8")

def _preprocessor(frame,d):
    numeric=list(frame.select_dtypes(include=["number","bool"]).columns)
    categorical=[c for c in frame.columns if c not in numeric]
    transformers=[]
    if numeric:
        transformers.append(("numeric",d["Pipeline"]([("impute",d["SimpleImputer"](strategy="median",add_indicator=True)),("scale",d["RobustScaler"]())]),numeric))
    if categorical:
        transformers.append(("categorical",d["Pipeline"]([("impute",d["SimpleImputer"](strategy="most_frequent")),("encode",d["OneHotEncoder"](handle_unknown="ignore"))]),categorical))
    return d["ColumnTransformer"](transformers,remainder="drop"),numeric,categorical

def _models(kind,n_classes,d):
    if kind=="classification":
        xgb_args={"objective":"multi:softprob","num_class":n_classes,"eval_metric":"mlogloss"} if n_classes>2 else {"objective":"binary:logistic","eval_metric":"logloss"}
        xgb=d["XGBClassifier"](**xgb_args,random_state=RANDOM_STATE,n_jobs=1)
        return {
          "logistic_regression":(d["LogisticRegression"](class_weight="balanced",max_iter=2000,random_state=RANDOM_STATE),{"model__C":[.1,1.0,10.0]}),
          "random_forest":(d["RandomForestClassifier"](class_weight="balanced",random_state=RANDOM_STATE,n_jobs=1),{"model__n_estimators":[50],"model__max_depth":[8,12],"model__min_samples_leaf":[3,8]}),
          "xgboost":(xgb,{"model__n_estimators":[75,100],"model__max_depth":[3,5],"model__learning_rate":[.05,.1],"model__subsample":[.8,1.0]})}
    return {
      "linear_regression":(d["LinearRegression"](),{"model__fit_intercept":[True,False]}),
      "random_forest":(d["RandomForestRegressor"](random_state=RANDOM_STATE,n_jobs=1),{"model__n_estimators":[50],"model__max_depth":[8,12],"model__min_samples_leaf":[3,8]}),
      "xgboost":(d["XGBRegressor"](objective="reg:squarederror",random_state=RANDOM_STATE,n_jobs=1),{"model__n_estimators":[75,100],"model__max_depth":[3,5],"model__learning_rate":[.05,.1],"model__subsample":[.8,1.0]})}

def _metrics(kind,y_true,y_pred,d,classes=None,probabilities=None):
    if kind=="classification":
        report=d["classification_report"](y_true,y_pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
        cm=d["confusion_matrix"](y_true,y_pred,labels=list(range(len(classes)))).tolist()
        result={"accuracy":float(d["accuracy_score"](y_true,y_pred)),
            "precision_macro":float(d["precision_score"](y_true,y_pred,average="macro",zero_division=0)),
            "precision_weighted":float(d["precision_score"](y_true,y_pred,average="weighted",zero_division=0)),
            "recall_macro":float(d["recall_score"](y_true,y_pred,average="macro",zero_division=0)),
            "recall_weighted":float(d["recall_score"](y_true,y_pred,average="weighted",zero_division=0)),
            "macro_f1":float(d["f1_score"](y_true,y_pred,average="macro",zero_division=0)),
            "weighted_f1":float(d["f1_score"](y_true,y_pred,average="weighted",zero_division=0)),
            "per_class":report,"confusion_matrix":cm,"classes":classes}
        try:
            if probabilities is not None:
                if len(classes)==2: auc=d["roc_auc_score"](y_true,probabilities[:,1])
                else: auc=d["roc_auc_score"](y_true,probabilities,multi_class="ovr",average="macro",labels=list(range(len(classes))))
                result["roc_auc_ovr_macro"]=float(auc)
                truth=d["np"].eye(len(classes))[y_true.astype(int)]
                result["multiclass_brier_score"]=float(d["np"].mean(d["np"].sum((probabilities-truth)**2,axis=1)))
                result["log_loss"]=float(d["log_loss"](y_true,probabilities,labels=list(range(len(classes)))))
                confidence=probabilities.max(axis=1); correct=(probabilities.argmax(axis=1)==y_true).astype(float)
                bins=[]; ece=0.0
                for lower in d["np"].linspace(0,1,11)[:-1]:
                    upper=lower+.1; mask=(confidence>=lower)&((confidence<upper) if upper<1 else (confidence<=upper))
                    if not mask.any(): continue
                    mean_conf=float(confidence[mask].mean()); accuracy=float(correct[mask].mean()); count=int(mask.sum())
                    bins.append({"lower":float(lower),"upper":float(upper),"count":count,"mean_confidence":mean_conf,"accuracy":accuracy})
                    ece+=count/len(correct)*abs(mean_conf-accuracy)
                result["expected_calibration_error_10_bins"]=float(ece)
                result["calibration_bins"]=bins
        except ValueError: pass
        return result
    residual=y_true-y_pred
    return {"mae":float(d["mean_absolute_error"](y_true,y_pred)),"mse":float(d["mean_squared_error"](y_true,y_pred)),
        "rmse":float(math.sqrt(d["mean_squared_error"](y_true,y_pred))),"r2":float(d["r2_score"](y_true,y_pred)),
        "residual_mean":float(residual.mean()),"residual_std":float(residual.std()),"residual_max_abs":float(abs(residual).max())}

def train_all(csv_path,tracking_uri="file:./mlruns",artifact_dir="artifacts/models",data_authorized=False,quick=False,max_rows=MAX_TRAIN_ROWS):
    """Train six models only after the caller affirms training data rights."""
    if not data_authorized: raise PermissionError("Confirm that you are authorized to use this dataset before training.")
    temp_root=Path(os.environ.get("AURELI_TEMP_DIR",Path(".tmp")/"ml-temp")).resolve()
    temp_root.mkdir(parents=True,exist_ok=True)
    os.environ["TEMP"]=str(temp_root); os.environ["TMP"]=str(temp_root); tempfile.tempdir=str(temp_root)
    d=_deps(); pd=d["pd"]; np=d["np"]; path=Path(csv_path)
    if not path.is_file(): raise FileNotFoundError(f"Dataset not found: {path}")
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    original=pd.read_csv(path,low_memory=False); validate_dataset(original)
    artifact_dir=Path(artifact_dir); artifact_dir.mkdir(parents=True,exist_ok=True)
    cached_profile=artifact_dir/"dataset_profile.json"
    if cached_profile.is_file():
        try:
            candidate=json.loads(cached_profile.read_text(encoding="utf-8"))
            profile=candidate if candidate.get("sha256")==fingerprint else inspect_dataset(path)
        except (OSError,ValueError): profile=inspect_dataset(path)
    else: profile=inspect_dataset(path)
    cleaned,format_repairs,unparsed,numeric_repairs=_clean_frame(original,pd)
    eligibility=cleaned[TARGETS["classification"]].astype("string").str.strip()
    eligibility=eligibility.replace({"":"<NA>","nan":"<NA>","None":"<NA>"})
    max_emi=pd.to_numeric(cleaned[TARGETS["regression"]],errors="coerce")
    duplicate=cleaned.duplicated()
    bad_class=eligibility.isna()|eligibility.eq("<NA>")
    bad_target=max_emi.isna()|~np.isfinite(max_emi)|(max_emi<0)
    usable=~duplicate & ~bad_class & ~bad_target
    profile["cleaning_decisions"]={"normalized_whitespace_and_category_variants":format_repairs,"repaired_numeric_serialization_suffixes":numeric_repairs,
        "unparsed_numeric_values_retained_as_missing_for_training_imputation":unparsed,
        "no financial values fabricated":True,"duplicate_policy":"drop exact duplicate rows after deterministic formatting normalization",
        "missing_feature_policy":"retain rows; median/mode imputation is fit on training folds only",
        "target_policy":"exclude rows with missing/invalid emi_eligibility or negative/non-numeric max_monthly_emi"}
    profile["excluded_records"]={"duplicates_after_cleaning":int(duplicate.sum()),
        "invalid_emi_eligibility_target":int((~duplicate & bad_class).sum()),
        "invalid_max_monthly_emi_target":int((~duplicate & ~bad_class & bad_target).sum())}
    data=cleaned.loc[usable].copy().reset_index(drop=True)
    profile["rows_retained_after_cleaning"]=int(len(data))
    if not len(data): raise ValueError("No rows remain after removing duplicates and records with invalid targets.")
    raw_id_cols=[c for c in data.columns if c.lower() in {"id","loan_id","customer_id","user_id","application_id"}]
    input_feature_columns=[c for c in data.columns if c not in TARGETS.values() and c not in raw_id_cols]
    raw_feature_data=data[input_feature_columns]
    typed_input=_engineer(raw_feature_data,pd)
    raw_numeric_features=[col for col in input_feature_columns if col in typed_input.select_dtypes(include=["number","bool"]).columns]
    raw_categorical_features=[c for c in input_feature_columns if c not in raw_numeric_features]
    input_categorical_values={col:sorted(str(x) for x in typed_input[col].dropna().unique())[:50] for col in raw_categorical_features}
    input_numeric_defaults={col:float(typed_input[col].median()) if typed_input[col].notna().any() else None for col in raw_numeric_features}
    if max_rows and len(data)>int(max_rows):
        bins=pd.qcut(data[TARGETS["regression"]],q=10,duplicates="drop").cat.codes.astype(str)
        strata=data[TARGETS["classification"]].astype(str)+"|"+bins
        try:
            data,_,=d["train_test_split"](data,train_size=int(max_rows),random_state=RANDOM_STATE,stratify=strata)
            profile["sampling_method"]="reproducible proportional stratification by eligibility class and max_monthly_emi quantile"
        except ValueError:
            data,_=d["train_test_split"](data,train_size=int(max_rows),random_state=RANDOM_STATE,stratify=data[TARGETS["classification"]])
            profile["sampling_method"]="reproducible proportional stratification by eligibility class (fallback)"
    else: profile["sampling_method"]="all valid authorized rows; no sampling required"
    data=data.reset_index(drop=True)
    profile["training_rows_used"]=int(len(data)); profile["valid_rows_not_sampled"] = int(profile["rows_retained_after_cleaning"]-len(data))
    profile["split_plan"]={"train":0.6,"validation":0.2,"test":0.2,"random_state":RANDOM_STATE,"cross_validation_folds":2}
    data=_engineer(data,pd)
    id_cols=[c for c in data.columns if c.lower() in {"id","loan_id","customer_id","user_id","application_id"}]
    xcols=[c for c in data.columns if c not in TARGETS.values() and c not in id_cols]
    features=data[xcols]
    d["mlflow"].set_tracking_uri(tracking_uri)
    results=[]
    (artifact_dir/"dataset_profile.json").write_text(json.dumps(profile,indent=2,default=str),encoding="utf-8")
    _write_quality_report(profile,artifact_dir/"data_quality_report.md")
    for kind,target in TARGETS.items():
        y=data[target]
        if kind=="classification":
            y=y.astype("string").str.strip().replace("",pd.NA)
            valid=y.notna() & ~y.isin(["nan","None"])
            encoder=d["LabelEncoder"](); y=pd.Series(encoder.fit_transform(y[valid]),index=y[valid].index)
            classes=[str(x) for x in encoder.classes_]
            counts=y.value_counts()
            if counts.min()<5: raise ValueError(f"Each {target} class needs at least five rows for train/validation/test and CV splits.")
        else:
            y=pd.to_numeric(y,errors="coerce"); valid=y.notna() & np.isfinite(y) & (y>=0)
            y=y[valid].astype(float); classes=None
        X=features.loc[valid].copy()
        if len(X)<30: raise ValueError(f"At least 30 valid records are required to train {target}.")
        stratify=y if kind=="classification" else None
        # Independent 60/20/20 train/validation/test split. Holdout test is never
        # used to tune or select the winning candidate.
        X_trainval,X_test,y_trainval,y_test=d["train_test_split"](X,y,test_size=.2,random_state=RANDOM_STATE,stratify=stratify)
        X_train,X_val,y_train,y_val=d["train_test_split"](X_trainval,y_trainval,test_size=.25,random_state=RANDOM_STATE,stratify=y_trainval if kind=="classification" else None)
        cv=d["StratifiedKFold"](n_splits=2,shuffle=True,random_state=RANDOM_STATE) if kind=="classification" else 2
        candidates=_models(kind,len(classes) if classes else 0,d); task_runs=[]
        experiment=f"aureli-{kind}-{target}"; d["mlflow"].set_experiment(experiment)
        for model_name,(estimator,grid) in candidates.items():
            prep,numeric,categorical=_preprocessor(X_train,d)
            pipe=d["Pipeline"]([("preprocess",prep),("model",estimator)])
            scoring="f1_macro" if kind=="classification" else "neg_root_mean_squared_error"
            quick_grid={key:values[:1] for key,values in grid.items()}
            search=(d["GridSearchCV"](pipe,quick_grid,cv=cv,scoring=scoring,n_jobs=1,refit=True,error_score="raise") if quick else
                    d["RandomizedSearchCV"](pipe,grid,n_iter=min(2,math.prod(len(v) for v in grid.values())),cv=cv,scoring=scoring,n_jobs=1,random_state=RANDOM_STATE,refit=True,error_score="raise"))
            fit_kwargs={}
            if kind=="classification" and model_name=="xgboost":
                fit_kwargs["model__sample_weight"]=d["compute_sample_weight"]("balanced",y_train)
            started=time.perf_counter()
            with d["mlflow"].start_run(run_name=model_name) as run:
                search.fit(X_train,y_train,**fit_kwargs)
                selected=search.best_estimator_
                val_pred=selected.predict(X_val)
                val_prob=selected.predict_proba(X_val) if kind=="classification" and hasattr(selected,"predict_proba") else None
                val_metric=_metrics(kind,np.asarray(y_val),np.asarray(val_pred),d,classes,val_prob)
                validation_score=val_metric["macro_f1"] if kind=="classification" else val_metric["rmse"]
                # Final refit includes train+validation and still does not touch test.
                final_prep,numeric,categorical=_preprocessor(X_trainval,d)
                final_pipe=d["Pipeline"]([("preprocess",final_prep),("model",selected.named_steps["model"])] )
                final_kwargs={}
                if kind=="classification" and model_name=="xgboost": final_kwargs["model__sample_weight"]=d["compute_sample_weight"]("balanced",y_trainval)
                final_pipe.fit(X_trainval,y_trainval,**final_kwargs)
                test_pred=final_pipe.predict(X_test)
                test_prob=final_pipe.predict_proba(X_test) if kind=="classification" and hasattr(final_pipe,"predict_proba") else None
                test_metrics=_metrics(kind,np.asarray(y_test),np.asarray(test_pred),d,classes,test_prob)
                scalar_metrics={k:v for k,v in test_metrics.items() if isinstance(v,(int,float))}
                d["mlflow"].log_params({"model":model_name,"target":target,"random_state":RANDOM_STATE,"cv_folds":2,
                    "split":"60_20_20","feature_count":len(xcols),"numeric_features":len(numeric),"categorical_features":len(categorical),
                    "training_rows":len(X_trainval),"validation_rows":len(X_val),"test_rows":len(X_test),
                    "valid_rows_not_sampled":profile["valid_rows_not_sampled"],"dataset_sha256":fingerprint,**search.best_params_})
                d["mlflow"].log_metric("cv_best_score",float(search.best_score_))
                d["mlflow"].log_metric("validation_selection_score",float(validation_score))
                d["mlflow"].log_metrics(scalar_metrics)
                d["mlflow"].set_tags({"dataset_sha256":fingerprint,"task":kind,"test_set":"untouched_20_percent","class_imbalance":"class_weight_balanced" if kind=="classification" else "not_applicable"})
                if classes: d["mlflow"].log_dict({str(i):name for i,name in enumerate(classes)},"class_labels.json")
                d["mlflow"].log_dict(test_metrics,"test_metrics.json")
                importance=_feature_importance(final_pipe,d)
                d["mlflow"].log_dict(importance,"feature_importance.json")
                evaluation_path=_log_evaluation_plots(kind,y_test,test_pred,test_metrics,classes,run.info.run_id,d,artifact_dir,target,model_name)
                _log_shap(final_pipe,X_test,d,run.info.run_id)
                d["mlflow_sklearn"].log_model(final_pipe,artifact_path="model")
                duration=time.perf_counter()-started
                d["mlflow"].log_metric("training_duration_seconds",float(duration))
                d["mlflow"].log_dict(profile,"data_quality/dataset_profile.json")
                row={"task":kind,"target":target,"model":model_name,"run_id":run.info.run_id,"experiment_id":run.info.experiment_id,
                     "cv_best_score":float(search.best_score_),"validation_score":float(validation_score),
                     "best_params":search.best_params_,"evaluation_path":evaluation_path,**scalar_metrics}
                results.append(row); task_runs.append({**row,"pipeline":final_pipe,"classes":classes,"features":xcols,"numeric":numeric,"categorical":categorical,"metric_payload":test_metrics,"importance":importance.get("features",[])})
        winner=(max if kind=="classification" else min)(task_runs,key=lambda row:row["validation_score"] if kind=="classification" else row["validation_score"])
        model_name=f"aureli-{target}"
        registered=d["mlflow"].register_model(f"runs:/{winner['run_id']}/model",model_name)
        artifact={"pipeline":winner["pipeline"],"target":target,"task":kind,"model_name":winner["model"],
            "classes":winner["classes"],"feature_columns":winner["features"],"numeric_features":winner["numeric"],
            "categorical_features":winner["categorical"],"dataset_sha256":fingerprint,"metrics":winner["metric_payload"],
            "validation_score":winner["validation_score"],"registered_model":registered.name,"registered_version":registered.version,
            "input_feature_columns":input_feature_columns,"input_numeric_features":raw_numeric_features,
            "input_categorical_values":input_categorical_values,"input_numeric_defaults":input_numeric_defaults,
            "feature_importance":winner["importance"],"evaluation_path":winner["evaluation_path"],
            "format_version":1,"random_state":RANDOM_STATE}
        if kind=="regression":
            artifact["target_min"]=float(data[target].min()); artifact["target_max"]=float(data[target].max())
        d["joblib"].dump(artifact,artifact_dir/f"{target}.joblib")
        (artifact_dir/f"{target}.metadata.json").write_text(json.dumps({k:v for k,v in artifact.items() if k!="pipeline"},indent=2,default=str),encoding="utf-8")
        results.append({k:winner[k] for k in ("task","target","model","run_id","validation_score")}|{"registered_model":registered.name,"registered_version":registered.version,"artifact":str(artifact_dir/f"{target}.joblib")})
    comparison=pd.DataFrame([{k:v for k,v in row.items() if k!="pipeline" and k!="metric_payload"}
                             for row in results if row.get("evaluation_path")])
    comparison.to_csv(artifact_dir/"model_comparison.csv",index=False)
    return results

def _feature_importance(pipeline,d):
    model=pipeline.named_steps["model"]; names=pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    if hasattr(model,"feature_importances_"): values=d["np"].abs(model.feature_importances_)
    elif hasattr(model,"coef_"):
        coefficients=d["np"].abs(model.coef_)
        values=coefficients.mean(axis=0) if coefficients.ndim>1 else coefficients
    else: return {"available":False,"features":[]}
    order=d["np"].argsort(values)[::-1][:50]
    return {"available":True,"features":[{"name":names[i],"importance":float(values[i])} for i in order]}

def _log_evaluation_plots(kind,y_true,pred,metrics,classes,run_id,d,artifact_dir,target,model_name):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    output=Path(artifact_dir)/"evaluation"/target/model_name
    output.mkdir(parents=True,exist_ok=True)
    if kind=="classification":
        fig=px.imshow(metrics["confusion_matrix"],x=classes,y=classes,text_auto=True,color_continuous_scale="Teal",labels={"x":"Predicted","y":"Actual","color":"Count"})
        name="confusion_matrix.html"
    else:
        import pandas as pd
        fig=go.Figure(); fig.add_scatter(x=y_true,y=pred,mode="markers",name="Predictions")
        low=min(float(min(y_true)),float(min(pred))); high=max(float(max(y_true)),float(max(pred)))
        fig.add_scatter(x=[low,high],y=[low,high],mode="lines",name="Perfect prediction")
        fig.update_layout(xaxis_title="Actual max_monthly_emi",yaxis_title="Predicted max_monthly_emi")
        name="actual_vs_predicted.html"
    path=output/name; fig.write_html(path); d["mlflow"].log_artifact(str(path),artifact_path="evaluation")
    if kind=="classification" and metrics.get("calibration_bins"):
        calibration=go.Figure()
        calibration.add_scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line={"dash":"dash"})
        calibration.add_scatter(x=[row["mean_confidence"] for row in metrics["calibration_bins"]],
            y=[row["accuracy"] for row in metrics["calibration_bins"]],mode="lines+markers",
            text=[f"n={row['count']}" for row in metrics["calibration_bins"]],name="Observed")
        calibration.update_layout(title="Eligibility probability calibration",xaxis_title="Mean predicted confidence",yaxis_title="Observed accuracy",xaxis_range=[0,1],yaxis_range=[0,1])
        cal_path=output/"calibration.html"; calibration.write_html(cal_path); d["mlflow"].log_artifact(str(cal_path),artifact_path="evaluation")
    if kind=="regression":
        residuals=y_true-pred
        data=pd.DataFrame({"actual":y_true,"predicted":pred,"residual":residuals})
        residual_path=output/"residuals.csv"; data.to_csv(residual_path,index=False)
        d["mlflow"].log_artifact(str(residual_path),artifact_path="evaluation")
        residual_fig=make_subplots(rows=1,cols=2,subplot_titles=("Residual distribution","Residual by predicted EMI"))
        residual_fig.add_histogram(x=residuals,nbinsx=30,row=1,col=1,name="Residuals")
        residual_fig.add_scatter(x=pred,y=residuals,mode="markers",row=1,col=2,name="Residual")
        residual_fig.add_hline(y=0,line_dash="dash",line_color="#3bc6a5",row=1,col=2)
        residual_fig.update_layout(title="Regression residual diagnostics",showlegend=False)
        residual_path=output/"residuals.html"; residual_fig.write_html(residual_path)
        d["mlflow"].log_artifact(str(residual_path),artifact_path="evaluation")
    return str(output)

def _log_shap(pipeline,X_test,d,run_id):
    import tempfile
    try:
        import shap
        import numpy as np
        transformed=pipeline.named_steps["preprocess"].transform(X_test.iloc[:min(64,len(X_test))])
        model=pipeline.named_steps["model"]
        explainer=shap.TreeExplainer(model)
        values=explainer.shap_values(transformed)
        if isinstance(values,list): values=np.stack(values,axis=-1)
        values=np.asarray(values)
        importance=np.abs(values).mean(axis=tuple(i for i in range(values.ndim) if i!=1))
        names=pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
        rows=sorted(({"feature":name,"mean_abs_shap":float(score)} for name,score in zip(names,importance)),key=lambda row:row["mean_abs_shap"],reverse=True)[:50]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"shap_global_importance.json"; path.write_text(json.dumps(rows,indent=2),encoding="utf-8")
            d["mlflow"].log_artifact(str(path),artifact_path="explainability")
    except Exception as exc:
        d["mlflow"].set_tag("shap_status",f"unavailable:{type(exc).__name__}")
