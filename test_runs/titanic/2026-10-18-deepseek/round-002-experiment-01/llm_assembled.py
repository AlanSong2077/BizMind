import os, json, joblib, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

SEED = 42
WARD = "work"; os.makedirs(WARD, exist_ok=True)
base = "../input/data"
train = pd.read_csv(f"{base}/train_data/train.csv")
val   = pd.read_csv(f"{base}/validation_data/validation.csv")
test  = pd.read_csv(f"{base}/predict_data/test.csv")

# ---- fixed validation IDs (protocol guard) ----
val_ids_path = f"{WARD}/val_ids.json"
val_ids = val["PassengerId"].tolist()
if os.path.exists(val_ids_path):
    assert json.load(open(val_ids_path)) == val_ids, "val IDs changed"
else:
    json.dump(val_ids, open(val_ids_path, "w"))
assert not set(train["PassengerId"]) & set(val["PassengerId"])
assert not set(train["PassengerId"]) & set(test["PassengerId"])
assert not set(val["PassengerId"]) & set(test["PassengerId"])

def featurize(df):
    d = df.copy()
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    title = d["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    d["Title"] = title.fillna("Rare")
    return d

tr_f, val_f, te_f = featurize(train), featurize(val), featurize(test)
tc = tr_f["Title"].value_counts()
rare = tc[tc < 10].index
for d in (tr_f, val_f, te_f):
    d["Title"] = d["Title"].where(~d["Title"].isin(rare), "Rare")
    d["Embarked"] = d["Embarked"].fillna("M")

num_cols = ["Pclass", "SibSp", "Parch", "FamilySize", "IsAlone"]
af_cols  = ["Age", "Fare"]
cat_cols = ["Sex", "Embarked", "Title"]

def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("af",  Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), af_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="M")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

y = tr_f["Survived"].values
X = tr_f[num_cols + af_cols + cat_cols]

# ---- incumbent config (best measured: fixed val 0.8492) ----
BW, BD, BL = 0.5, 6, 4
lr_final = Pipeline([("pre", make_pre()), ("lr", LogisticRegression(max_iter=2000, random_state=SEED))]).fit(X, y)
et_final = Pipeline([("pre", make_pre()), ("et", ExtraTreesClassifier(
    n_estimators=400, max_depth=BD, min_samples_leaf=BL,
    random_state=SEED, n_jobs=4))]).fit(X, y)

Xv = val_f[num_cols + af_cols + cat_cols]
Xt = te_f[num_cols + af_cols + cat_cols]
def blend(Xf):
    return BW * lr_final.predict_proba(Xf)[:, 1] + (1 - BW) * et_final.predict_proba(Xf)[:, 1]

val_pred  = (blend(Xv) >= 0.5).astype(int)
test_pred = (blend(Xt) >= 0.5).astype(int)

pd.DataFrame({"PassengerId": val["PassengerId"], "Survived": val_pred}).to_csv(f"{WARD}/val_predictions.csv", index=False)
pp = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
pp.to_csv("predictions.csv", index=False)

metrics = {
    "seed": SEED,
    "round": 5,
    "decision": "reverted to incumbent; round-4 CV sweep (0.8436 val) did NOT beat 0.8492 -> negative result recorded",
    "chosen": {"w": BW, "depth": BD, "leaf": BL},
    "incumbent_fixed_val_accuracy": 0.8492,
    "round4_cv_tuned_fixed_val_accuracy": 0.8436,
    "note": "prob>=0.5 -> 1; both val/test predictions from same incumbent model; fixed-validation acc scored by evaluator",
}
json.dump(metrics, open(f"{WARD}/metrics.json", "w"), indent=2)
joblib.dump({"lr": lr_final, "et": et_final, "w": BW, "pre": "shared ColumnTransformer embedded in each pipeline"}, f"{WARD}/model_blend_best.joblib")

open(f"{WARD}/README.txt", "w").write(
    "ROUND 5 - reverted to incumbent LR+ExtraTrees soft-vote (best measured)\n"
    f"Config: w={BW}, ET max_depth={BD}, min_samples_leaf={BL}. Fixed-val accuracy 0.8492 (incumbent).\n"
    "Round-4 CV sweep (w=0.4,d=6,leaf=3) gave fixed-val 0.8436 -> did not beat incumbent; reverted.\n"
    "Inference: prob = 0.5*p_LR + 0.5*p_ET; Survived=1 if prob>=0.5 else 0.\n"
    "Fit on all 712 train_data/train.csv rows; validation/test transform-only. Seed 42, CPU only.\n"
    "Deliverables: predictions.csv (418 rows), work/val_predictions.csv (179 rows) from same model.\n")

# ---- reproducibility + schema assertions ----
vp = pd.read_csv(f"{WARD}/val_predictions.csv")
assert list(vp.columns) == ["PassengerId", "Survived"] and len(vp) == 179
assert vp["PassengerId"].tolist() == val_ids
assert list(pp.columns) == ["PassengerId", "Survived"] and len(pp) == 418
assert pp["PassengerId"].tolist() == test["PassengerId"].tolist()
assert set(vp["Survived"].unique()) <= {0, 1} and set(pp["Survived"].unique()) <= {0, 1}

# reload reproducibility
loaded = joblib.load(f"{WARD}/model_blend_best.joblib")
_va = (loaded["w"] * loaded["lr"].predict_proba(Xv)[:, 1] + (1 - loaded["w"]) * loaded["et"].predict_proba(Xv)[:, 1] >= 0.5).astype(int)
_te = (loaded["w"] * loaded["lr"].predict_proba(Xt)[:, 1] + (1 - loaded["w"]) * loaded["et"].predict_proba(Xt)[:, 1] >= 0.5).astype(int)
assert (_va == vp["Survived"].values).all(), "val drift"
assert (_te == pp["Survived"].values).all(), "test drift"

print("Reverted to incumbent w=%.1f depth=%d leaf=%d (fixed-val 0.8492)" % (BW, BD, BL))
print("val pos rate %.3f | test pos rate %.3f" % (vp["Survived"].mean(), pp["Survived"].mean()))
print("Reproducibility OK. Deliverables: predictions.csv (418) + work/val_predictions.csv (179)")