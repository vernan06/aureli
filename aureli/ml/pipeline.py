"""Reproducible EMI classification/regression with MLflow and local artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

TARGETS={"classification":"emi_eligibility","regression":"max_monthly_emi"}
RANDOM_STATE=42

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
    from sklearn.metrics import (accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,
        f1_score,mean_absolute_error,mean_squared_error,precision_score,recall_score,r2_score,roc_auc_score)
    from sklearn.model_selection import (GridSearchCV,RandomizedSearchCV,StratifiedKFold,train_test_split)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder,RobustScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier,XGBRegressor
    dependencies=locals()
    dependencies["mlflow_sklearn"]=mlflow.sklearn
    return dependencies

def validate_dataset(frame):
    required=set(TARGETS.values())
    missing=required-set(frame.columns)
    if missing: raise ValueError("Missing required target column(s): "+", ".join(sorted(missing)))
    if len(frame)==0: raise ValueError("Dataset contains no rows.")
    feature_columns=[c for c in frame.columns if c not in required and c.lower() not in {"id","loan_id","customer_id","user_id","application_id"}]
    if not feature_columns: raise ValueError("Dataset must contain at least one input feature.")
    return feature_columns

def _engineer(frame, pd):
    data=frame.copy()
    data.columns=[str(c).strip() for c in data.columns]
    # Numeric-looking text is parsed only when at least 80% of its non-empty
    # values parse as numbers. Ambiguous columns remain categorical.
    for col in data.select_dtypes(include=["object","string"]).columns:
        if col in TARGETS.values(): continue
        numeric=pd.to_numeric(data[col],errors="coerce")
        nonempty=int(data[col].notna().sum())
        if nonempty and int(numeric.notna().sum())>=math.ceil(.8*nonempty): data[col]=numeric
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
    return data

def inspect_dataset(csv_path):
    d=_deps(); path=Path(csv_path); frame=d["pd"].read_csv(path,low_memory=False)
    features=validate_dataset(frame); numeric=frame[features].select_dtypes(include=["number"])
    outliers={}
    for col in numeric:
        series=numeric[col].dropna()
        if series.empty: continue
        q1,q3=series.quantile(.25),series.quantile(.75); spread=q3-q1
        count=int(((series<q1-1.5*spread)|(series>q3+1.5*spread)).sum()) if spread else 0
        if count: outliers[col]=count
    eligibility=frame[TARGETS["classification"]].dropna().astype(str).value_counts().to_dict()
    return {"rows":int(len(frame)),"columns":list(frame.columns),"feature_count":len(features),
        "duplicates":int(frame.duplicated().sum()),"missing":{str(k):int(v) for k,v in frame.isna().sum().items() if v},
        "outlier_counts_iqr":outliers,"eligibility_distribution":eligibility}

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
          "random_forest":(d["RandomForestClassifier"](class_weight="balanced",random_state=RANDOM_STATE,n_jobs=1),{"model__n_estimators":[100,200],"model__max_depth":[None,12],"model__min_samples_leaf":[1,3]}),
          "xgboost":(xgb,{"model__n_estimators":[100,200],"model__max_depth":[3,6],"model__learning_rate":[.03,.1],"model__subsample":[.8,1.0]})}
    return {
      "linear_regression":(d["LinearRegression"](),{"model__fit_intercept":[True,False]}),
      "random_forest":(d["RandomForestRegressor"](random_state=RANDOM_STATE,n_jobs=1),{"model__n_estimators":[100,200],"model__max_depth":[None,12],"model__min_samples_leaf":[1,3]}),
      "xgboost":(d["XGBRegressor"](objective="reg:squarederror",random_state=RANDOM_STATE,n_jobs=1),{"model__n_estimators":[100,200],"model__max_depth":[3,6],"model__learning_rate":[.03,.1],"model__subsample":[.8,1.0]})}

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
        except ValueError: pass
        return result
    residual=y_true-y_pred
    return {"mae":float(d["mean_absolute_error"](y_true,y_pred)),"mse":float(d["mean_squared_error"](y_true,y_pred)),
        "rmse":float(math.sqrt(d["mean_squared_error"](y_true,y_pred))),"r2":float(d["r2_score"](y_true,y_pred)),
        "residual_mean":float(residual.mean()),"residual_std":float(residual.std()),"residual_max_abs":float(abs(residual).max())}

def train_all(csv_path,tracking_uri="file:./mlruns",artifact_dir="models",data_authorized=False,quick=False):
    """Train six models only after the caller affirms training data rights."""
    if not data_authorized: raise PermissionError("Confirm that you are authorized to use this dataset before training.")
    temp_root=Path(os.environ.get("AURELI_TEMP_DIR",Path(".tmp")/"ml-temp")).resolve()
    temp_root.mkdir(parents=True,exist_ok=True)
    os.environ["TEMP"]=str(temp_root); os.environ["TMP"]=str(temp_root); tempfile.tempdir=str(temp_root)
    d=_deps(); pd=d["pd"]; np=d["np"]; path=Path(csv_path)
    if not path.is_file(): raise FileNotFoundError(f"Dataset not found: {path}")
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=pd.read_csv(path,low_memory=False); validate_dataset(raw)
    profile=inspect_dataset(path)
    raw=raw.drop_duplicates().reset_index(drop=True)
    raw_id_cols=[c for c in raw.columns if c.lower() in {"id","loan_id","customer_id","user_id","application_id"}]
    input_feature_columns=[c for c in raw.columns if c not in TARGETS.values() and c not in raw_id_cols]
    raw_feature_data=raw[input_feature_columns]
    typed_input=_engineer(raw_feature_data,pd)
    raw_numeric_features=[col for col in input_feature_columns if col in typed_input.select_dtypes(include=["number","bool"]).columns]
    raw_categorical_features=[c for c in input_feature_columns if c not in raw_numeric_features]
    input_categorical_values={col:sorted(str(x) for x in typed_input[col].dropna().unique())[:50] for col in raw_categorical_features}
    input_numeric_defaults={col:float(typed_input[col].median()) if typed_input[col].notna().any() else None for col in raw_numeric_features}
    data=_engineer(raw,pd)
    id_cols=[c for c in data.columns if c.lower() in {"id","loan_id","customer_id","user_id","application_id"}]
    xcols=[c for c in data.columns if c not in TARGETS.values() and c not in id_cols]
    features=data[xcols]
    d["mlflow"].set_tracking_uri(tracking_uri)
    results=[]; artifact_dir=Path(artifact_dir); artifact_dir.mkdir(parents=True,exist_ok=True)
    (artifact_dir/"dataset_profile.json").write_text(json.dumps(profile,indent=2,default=str),encoding="utf-8")
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
        cv=d["StratifiedKFold"](n_splits=3,shuffle=True,random_state=RANDOM_STATE) if kind=="classification" else 3
        candidates=_models(kind,len(classes) if classes else 0,d); task_runs=[]
        experiment=f"aureli-{kind}-{target}"; d["mlflow"].set_experiment(experiment)
        for model_name,(estimator,grid) in candidates.items():
            prep,numeric,categorical=_preprocessor(X_train,d)
            pipe=d["Pipeline"]([("preprocess",prep),("model",estimator)])
            scoring="f1_macro" if kind=="classification" else "neg_root_mean_squared_error"
            quick_grid={key:values[:1] for key,values in grid.items()}
            search=(d["GridSearchCV"](pipe,quick_grid,cv=cv,scoring=scoring,n_jobs=1,refit=True,error_score="raise") if quick else
                    d["RandomizedSearchCV"](pipe,grid,n_iter=min(6,math.prod(len(v) for v in grid.values())),cv=cv,scoring=scoring,n_jobs=1,random_state=RANDOM_STATE,refit=True,error_score="raise"))
            fit_kwargs={}
            if kind=="classification" and model_name=="xgboost":
                fit_kwargs["model__sample_weight"]=d["compute_sample_weight"]("balanced",y_train)
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
                d["mlflow"].log_params({"model":model_name,"target":target,"random_state":RANDOM_STATE,"cv_folds":3,
                    "split":"60_20_20","feature_count":len(xcols),"numeric_features":len(numeric),"categorical_features":len(categorical),"duplicate_rows_removed":profile["duplicates"],**search.best_params_})
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
        d["joblib"].dump(artifact,artifact_dir/f"{target}.joblib")
        (artifact_dir/f"{target}.metadata.json").write_text(json.dumps({k:v for k,v in artifact.items() if k!="pipeline"},indent=2,default=str),encoding="utf-8")
        results.append({k:winner[k] for k in ("task","target","model","run_id","validation_score")}|{"registered_model":registered.name,"registered_version":registered.version,"artifact":str(artifact_dir/f"{target}.joblib")})
    comparison=pd.DataFrame([{k:v for k,v in row.items() if k!="pipeline" and k!="metric_payload"} for row in results])
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
