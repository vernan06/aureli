"""Validated inference from Aureli's locally saved model pipelines."""
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

def artifact_path(target,model_dir="artifacts/models"):
    if target not in {"emi_eligibility","max_monthly_emi"}: raise ValueError("Unknown Aureli prediction target.")
    return Path(model_dir)/f"{target}.joblib"

def load_artifact(target,model_dir="artifacts/models"):
    path=artifact_path(target,model_dir)
    if not path.is_file(): raise FileNotFoundError(f"No trained Aureli artifact exists for {target}.")
    artifact=_load_cached(str(path.resolve()),path.stat().st_mtime_ns)
    if artifact.get("target")!=target or not hasattr(artifact.get("pipeline"),"predict"):
        raise ValueError("Saved model artifact is invalid or has a mismatched target.")
    return artifact

@lru_cache(maxsize=4)
def _load_cached(path,modified_ns):
    """Load immutable local pipelines once per process, invalidating on artifact change."""
    return joblib.load(path)

def predict(target,inputs,model_dir="artifacts/models"):
    artifact=load_artifact(target,model_dir)
    input_columns=artifact.get("input_feature_columns",artifact["feature_columns"])
    missing=[name for name in input_columns if name not in inputs]
    if missing: raise ValueError("Missing prediction fields: "+", ".join(missing))
    row={name:inputs[name] for name in input_columns}
    for name in artifact.get("input_numeric_features",artifact["numeric_features"]):
        value=row.get(name)
        if value in (None,""): continue
        try: numeric=float(value)
        except (TypeError,ValueError) as exc: raise ValueError(f"{name} must be numeric.") from exc
        if not np.isfinite(numeric): raise ValueError(f"{name} must be finite.")
        row[name]=numeric
    from aureli.ml.pipeline import _engineer
    frame=_engineer(pd.DataFrame([row],columns=input_columns),pd)
    frame=frame.reindex(columns=artifact["feature_columns"])
    pipeline=artifact["pipeline"]
    value=pipeline.predict(frame)[0]
    if target=="emi_eligibility":
        classes=artifact["classes"] or []
        index=int(value)
        if index<0 or index>=len(classes): raise ValueError("Model returned a class not present in its training metadata.")
        probs=pipeline.predict_proba(frame)[0] if hasattr(pipeline,"predict_proba") else None
        probabilities={classes[i]:float(probs[i]) for i in range(len(classes))} if probs is not None else None
        confidence=float(probs[index]) if probs is not None else None
        return {"target":target,"prediction":classes[index],"confidence":confidence,"probabilities":probabilities,
                "model_name":artifact["model_name"],"dataset_sha256":artifact["dataset_sha256"]}
    numeric=float(value)
    if not np.isfinite(numeric): raise ValueError("Model returned a non-finite affordable EMI value.")
    low=float(artifact.get("target_min",0.0)); high=float(artifact.get("target_max",float("inf")))
    bounded=min(max(numeric,low),high)
    return {"target":target,"prediction":bounded,"unbounded_prediction":numeric,"range_limited":bounded!=numeric,
            "observed_target_range":[low,high],"model_name":artifact["model_name"],"dataset_sha256":artifact["dataset_sha256"]}

def global_explanations(target,model_dir="artifacts/models",limit=20):
    artifact=load_artifact(target,model_dir)
    # The model's registered metadata includes permutation-independent tree or
    # linear importance computed on the transformed training features.
    path=Path(model_dir)/f"{target}.metadata.json"
    import json
    meta=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return meta.get("feature_importance",[])[:limit]

def explain_prediction(target,inputs,model_dir="artifacts/models",limit=12):
    artifact=load_artifact(target,model_dir)
    input_columns=artifact.get("input_feature_columns",artifact["feature_columns"])
    from aureli.ml.pipeline import _engineer
    frame=_engineer(pd.DataFrame([{name:inputs.get(name) for name in input_columns}],columns=input_columns),pd)
    frame=frame.reindex(columns=artifact["feature_columns"])
    pipeline=artifact["pipeline"]; model=pipeline.named_steps["model"]
    transformed=pipeline.named_steps["preprocess"].transform(frame)
    names=pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    try:
        if hasattr(model,"feature_importances_"):
            import shap
            values=shap.TreeExplainer(model).shap_values(transformed)
            if isinstance(values,list):
                prediction=int(pipeline.predict(frame)[0]); values=values[prediction]
            values=np.asarray(values)
            if values.ndim==3:
                prediction=int(pipeline.predict(frame)[0]); values=values[0,:,prediction]
            else: values=values.reshape(-1,values.shape[-1])[0]
            return {"method":"SHAP","features":sorted(({"feature":name,"contribution":float(value)} for name,value in zip(names,values)),key=lambda x:abs(x["contribution"]),reverse=True)[:limit]}
        if hasattr(model,"coef_"):
            coefficients=np.asarray(model.coef_)
            prediction=int(pipeline.predict(frame)[0]) if target=="emi_eligibility" else 0
            coefficients=coefficients[prediction] if coefficients.ndim==2 and prediction<coefficients.shape[0] else coefficients.reshape(-1)
            row=transformed.toarray()[0] if hasattr(transformed,"toarray") else np.asarray(transformed)[0]
            return {"method":"Linear model contributions","features":sorted(({"feature":name,"contribution":float(row[i]*coefficients[i])} for i,name in enumerate(names)),key=lambda x:abs(x["contribution"]),reverse=True)[:limit]}
    except (ImportError,ValueError,TypeError,AttributeError,IndexError):
        pass
    return {"method":"Global model importance only","features":global_explanations(target,model_dir,limit)}
