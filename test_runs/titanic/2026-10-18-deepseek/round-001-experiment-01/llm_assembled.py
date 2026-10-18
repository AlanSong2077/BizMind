import os, json, joblib, numpy as np, pandas as pd, warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

seed = 42
base = "../input/data"
train = pd.read_csv(f"{base}/train_data/train.csv")
val = pd.read_csv(f"{base}/validation_data/validation.csv")
test = pd.read_csv(f"{base}/predict_data/test.csv")

def featurize(df):
    d = df.copy()
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    title = d["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
    title = title.replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
    rare = ~title.isin(["Mr","Mrs","Miss","Master"])
    d["Title"] = np.where(rare, "Rare", title.fillna("Rare"))
    return d

tr_f, val_f, te_f = featurize(train), featurize(val), featurize(test)
X_tr = tr_f.drop(columns=["Survived"]); y_tr = tr_f["Survived"].values
val_ids = val_f["PassengerId"].tolist()

num_cols = ["Pclass", "SibSp", "Parch", "FamilySize", "IsAlone"]
af_cols = ["Age", "Fare"]
cat_cols = ["Sex", "Embarked", "Title"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("af", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), af_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="M")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, random_state=seed))])
clf.fit(X_tr, y_tr)

val_prob = clf.predict_proba(val_f)[:, 1]
val_pred = (val_prob >= 0.5).astype(int)
pd.DataFrame({"PassengerId": val_ids, "Survived": val_pred}).to_csv(
    "work/val_predictions.csv", index=False)

test_pred = (clf.predict_proba(te_f)[:, 1] >= 0.5).astype(int)
pp = pd.DataFrame({"PassengerId": te_f["PassengerId"].values, "Survived": test_pred})
pp.to_csv("predictions.csv", index=False)

json.dump({"model": "LR + Title/FamilySize/IsAlone (final best)",
           "seed": seed,
           "features_numeric": num_cols,
           "features_agefare": af_cols,
           "features_cat": cat_cols,
           "note": "best measured fixed-split validation accuracy across trials"},
          open("work/metrics.json", "w"), indent=2)
joblib.dump(clf, "work/model_final.joblib")
open("work/README.txt", "w").write(
    "FINAL INFERENCE NOTE\n"
    "Model: LogisticRegression on engineered Titanic features.\n"
    "Load work/model_final.joblib; predict_proba[:,1] >= 0.5 -> Survived=1 else 0.\n"
    "Features: scaled Pclass,SibSp,Parch,FamilySize(=SibSp+Parch+1),IsAlone; "
    "Age & Fare median-imputed with missing indicators then scaled; "
    "Sex, Embarked (missing->'M'), Title (Mr/Mrs/Miss/Master/rare->Rare) one-hot.\n"
    "Trained on all 712 rows of train_data/train.csv. Validation = 179 rows of "
    "validation_data/validation.csv aligned by PassengerId. Seed 42, CPU only.\n"
    "Deliverables: predictions.csv (418 rows, PassengerId+Survived), "
    "work/val_predictions.csv (179 rows).")

vp = pd.read_csv("work/val_predictions.csv")
assert vp["PassengerId"].tolist() == val_ids and len(vp) == 179
assert list(vp.columns) == ["PassengerId", "Survived"]
assert len(pp) == 418 and list(pp.columns) == ["PassengerId", "Survived"]
assert pp["PassengerId"].tolist() == te_f["PassengerId"].tolist()
assert set(pp["Survived"].unique()) <= {0, 1} and pp["Survived"].dtype.kind in "iu"
print("FINAL val rows", len(vp), "test rows", len(pp),
      "val pos rate", vp["Survived"].mean().round(3),
      "test pos rate", pp["Survived"].mean().round(3))
print(open("work/README.txt").read())

clf = joblib.load("work/model_final.joblib")

def featurize(df):
    d = df.copy()
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    title = d["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
    title = title.replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
    rare = ~title.isin(["Mr","Mrs","Miss","Master"])
    d["Title"] = np.where(rare, "Rare", title.fillna("Rare"))
    return d

_re_val = (clf.predict_proba(featurize(val))[:, 1] >= 0.5).astype(int)
_re_test = (clf.predict_proba(featurize(test))[:, 1] >= 0.5).astype(int)

vp = pd.read_csv("work/val_predictions.csv")
pp = pd.read_csv("predictions.csv")

# schema + alignment checks
assert list(vp.columns) == ["PassengerId", "Survived"] and len(vp) == 179
assert list(pp.columns) == ["PassengerId", "Survived"] and len(pp) == 418
assert vp["PassengerId"].tolist() == val["PassengerId"].tolist()
assert pp["PassengerId"].tolist() == test["PassengerId"].tolist()
assert set(vp["Survived"].unique()) <= {0,1} and set(pp["Survived"].unique()) <= {0,1}

# reproducibility: saved files equal fresh reload predictions
assert (vp["Survived"].values == _re_val).all(), "val drift"
assert (pp["Survived"].values == _re_test).all(), "test drift"

print("Reproducibility OK. Saved predictions match reloaded model.")
print("val rows", len(vp), "| test rows", len(pp))
print("metrics.json:\n", open("work/metrics.json").read())
print("Deliverables confirmed: predictions.csv + work/val_predictions.csv")