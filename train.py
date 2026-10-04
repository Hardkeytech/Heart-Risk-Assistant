"""Train on Cleveland (UCI) and report BOTH protocols honestly:
 1) stratified 80/20 holdout with a fixed seed (n=61 test patients)
 2) 5-fold x 20-repeat cross-validation (more reliable on 302 patients).
"""
import json, joblib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

SEED = 42
d = pd.read_csv("heart.csv").drop_duplicates()
y = 1 - d["target"]                      # source CSV: 0 = disease -> flip so 1 = disease
X = d.drop(columns="target")
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUM = [c for c in X if c not in CAT]

def pre():
    return ColumnTransformer([
        ("n", StandardScaler(), NUM),
        ("c", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])

def model():
    return Pipeline([("p", pre()), ("m", LogisticRegression(C=0.3, max_iter=3000))])

Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
h = model().fit(Xt, yt); pr = h.predict_proba(Xe)[:, 1]; pd_ = (pr >= .5).astype(int)
holdout = dict(accuracy=accuracy_score(ye, pd_), roc_auc=roc_auc_score(ye, pr), recall=recall_score(ye, pd_),
               precision=precision_score(ye, pd_), n_test=len(ye))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)
r = cross_validate(model(), X, y, cv=cv, scoring=["accuracy", "roc_auc", "recall", "precision"], n_jobs=-1)
cvm = {k: float(np.mean(r["test_" + k])) for k in ["accuracy", "roc_auc", "recall", "precision"]}
cvm["accuracy_std"] = float(np.std(r["test_accuracy"]))
metrics = dict(holdout={k: round(v, 3) for k, v in holdout.items()}, cv={k: round(v, 3) for k, v in cvm.items()}, n=len(d), seed=SEED)
print(json.dumps(metrics, indent=1))
joblib.dump(dict(model=model().fit(X, y), explainer=model().fit(X, y)), "model_bundle.joblib")
json.dump(metrics, open("metrics.json", "w"))
