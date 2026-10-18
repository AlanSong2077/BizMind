import os, json, joblib, numpy as np, pandas as pd, warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
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
    t = d["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
    t = t.replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
    d["Title"] = np.where(~t.isin(["Mr","Mrs","Miss","Master"]), "Rare", t.fillna("Rare"))
    return d

tr_f, val_f, te_f = featurize(train), featurize(val), featurize(test)
X_tr = tr_f.drop(columns=["Survived"]); y_tr = tr_f["Survived"].values
val_ids = val_f["PassengerId"].tolist()

num_cols = ["Pclass","SibSp","Parch","FamilySize","IsAlone"]
af_cols = ["Age","Fare"]
cat_cols = ["Sex","Embarked","Title"]

def pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num_cols),
        ("af", Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("sc",StandardScaler())]), af_cols),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="M")),
                          ("oh",OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

lr = Pipeline([("pre",pre()),("lr",LogisticRegression(max_iter=2000,random_state=seed))])
et = Pipeline([("pre",pre()),("et",ExtraTreesClassifier(n_estimators=400,max_depth=6,
                min_samples_leaf=4,random_state=seed,n_jobs=4))])
lr.fit(X_tr,y_tr); et.fit(X_tr,y_tr)

def blend(df):
    return 0.5*lr.predict_proba(df)[:,1] + 0.5*et.predict_proba(df)[:,1]

val_pred = (blend(val_f) >= 0.5).astype(int)
pd.DataFrame({"PassengerId":val_ids,"Survived":val_pred}).to_csv("work/val_predictions.csv", index=False)
test_pred = (blend(te_f) >= 0.5).astype(int)
pp = pd.DataFrame({"PassengerId":te_f["PassengerId"].values,"Survived":test_pred})
pp.to_csv("predictions.csv", index=False)

json.dump({"model":"0.5*LR + 0.5*ExtraTrees","seed":seed}, open("work/metrics.json","w"), indent=2)
joblib.dump({"lr":lr,"et":et}, "work/model_blend_et.joblib")
open("work/README.txt","w").write(
    "Inference: load model_blend_et.joblib; prob=0.5*lr.predict_proba+0.5*et.predict_proba; "
    ">=0.5 -> 1. Features as best LR. Seed 42, CPU, n_jobs=4.")

vp = pd.read_csv("work/val_predictions.csv")
assert vp["PassengerId"].tolist()==val_ids and len(vp)==179
assert len(pp)==418 and pp["PassengerId"].tolist()==te_f["PassengerId"].tolist()
assert set(pp["Survived"].unique())<={0,1}
print("val pos", vp["Survived"].mean().round(3), "test pos", pp["Survived"].mean().round(3))
