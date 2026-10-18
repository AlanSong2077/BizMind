import os, json, hashlib, joblib, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
warnings.filterwarnings("ignore")

SEED = 42
WARD = "work"; os.makedirs(WARD, exist_ok=True)
base = "../input/data"
train = pd.read_csv(f"{base}/train_data/train.csv")
val   = pd.read_csv(f"{base}/validation_data/validation.csv")
test  = pd.read_csv(f"{base}/predict_data/test.csv")
assert list(train.columns[:11])  # schema present
assert len(train) == 712 and len(val) == 179 and len(test) == 418

# ---- fixed validation ID protocol guard ----
val_ids_path = f"{WARD}/val_ids.json"
val_ids = val["PassengerId"].tolist()
if os.path.exists(val_ids_path):
    assert json.load(open(val_ids_path)) == val_ids, "val IDs changed"
else:
    json.dump(val_ids, open(val_ids_path, "w"))
assert not set(train["PassengerId"]) & set(val["PassengerId"])
assert not set(train["PassengerId"]) & set(test["PassengerId"])
assert not set(val["PassengerId"]) & set(test["PassengerId"])

# ---- featurize: Title-counts + rare mapping fit on train only ----
def base_featurize(df):
    d = df.copy()
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"]    = (d["FamilySize"] == 1).astype(int)
    t = d["Name"].str.extract(r",\s*([^.]+)\.", expand=False)
    t = t.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    t = t.fillna("Rare")
    # Deck = first Cabin letter, missing -> 'U'
    deck = d["Cabin"].astype("string").str[0]
    d["Deck"] = deck.where(deck.notna(), "U").fillna("U").astype(str)
    d["Embarked"] = d["Embarked"].fillna("M")
    d["_Title"] = t
    return d

tr_f  = base_featurize(train)
val_f = base_featurize(val)
te_f  = base_featurize(test)
tc = tr_f["_Title"].value_counts()
rare = tc[tc < 10].index
for d in (tr_f, val_f, te_f):
    d["Title"] = d["_Title"].where(~d["_Title"].isin(rare), "Rare")

num_cols = ["Pclass", "SibSp", "Parch", "FamilySize", "IsAlone"]
af_cols  = ["Age", "Fare"]

def make_pre(use_deck):
    cat_cols = ["Sex", "Embarked", "Title"] + (["Deck"] if use_deck else [])
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("af",  Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), af_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="M")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ]), cat_cols

def build_fit(use_deck):
    pre, cats = make_pre(use_deck)
    cols = num_cols + af_cols + cats
    X = tr_f[cols]; y = tr_f["Survived"].astype(int).values
    lr = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, random_state=SEED))]).fit(X, y)
    et = Pipeline([("pre", make_pre(use_deck)[0]), ("et", ExtraTreesClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=4, criterion="gini",
            random_state=SEED, n_jobs=4))]).fit(X, y)
    return lr, et, cols

def predict(lr, et, df, cols, w=0.5, thr=0.5):
    Xf = df[cols]
    p = w*lr.predict_proba(Xf)[:,1] + (1-w)*et.predict_proba(Xf)[:,1]
    return (p >= thr).astype(int)

# ---- incumbent (no Deck) and candidate (with Deck) ----
lr_i, et_i, cols_i = build_fit(False)
lr_d, et_d, cols_d = build_fit(True)

val_pred_inc = predict(lr_i, et_i, val_f, cols_i)
val_pred_dec = predict(lr_d, et_d, val_f, cols_d)
test_pred_inc = predict(lr_i, et_i, te_f, cols_i)
test_pred_dec = predict(lr_d, et_d, te_f, cols_d)

n_flip_val  = int((val_pred_inc != val_pred_dec).sum())
n_flip_test = int((test_pred_inc != test_pred_dec).sum())

# ---- decision policy ----
# I cannot read validation labels; incumbent is proven 0.8492. Deck is an unverified
# candidate. Deliver incumbent as the safe baseline, log Deck trial + its val CSV for
# independent scoring. (Selection happens on evaluator-reported accuracy.)
CHOSEN = "incumbent_no_deck"
val_pred  = val_pred_inc
test_pred = test_pred_inc

pp = pd.DataFrame({"PassengerId": test["PassengerId"].values, "Survived": test_pred.astype(int)})
pp.to_csv("predictions.csv", index=False)
vp = pd.DataFrame({"PassengerId": val["PassengerId"].values, "Survived": val_pred.astype(int)})
vp.to_csv(f"{WARD}/val_predictions.csv", index=False)
# also export the Deck candidate's val predictions for independent scoring/comparison
pd.DataFrame({"PassengerId": val["PassengerId"].values, "Survived": val_pred_dec.astype(int)}) \
    .to_csv(f"{WARD}/val_predictions_deck.csv", index=False)

joblib.dump({"lr": lr_i, "et": et_i, "w": 0.5, "threshold": 0.5, "use_deck": False,
             "cols": cols_i, "title_rare": list(rare)}, f"{WARD}/model_best.joblib")
joblib.dump({"lr": lr_d, "et": et_d, "w": 0.5, "threshold": 0.5, "use_deck": True,
             "cols": cols_d, "title_rare": list(rare)}, f"{WARD}/model_deck_test.joblib")

# ---- comparison ledger ----
comp_path = f"{WARD}/comparison.csv"
row = {"round": "deck", "variant": "LR+ET+Deck", "holdout_acc": "scored_by_evaluator",
       "accepted": "pending", "val_flips_vs_incumbent": n_flip_val, "test_flips_vs_incumbent": n_flip_test}
if os.path.exists(comp_path):
    cmp = pd.read_csv(comp_path)
    cmp = cmp[cmp["variant"] != "LR+ET+Deck"]
    cmp = pd.concat([cmp, pd.DataFrame([row])], ignore_index=True)
else:
    cmp = pd.DataFrame([row])
cmp.to_csv(comp_path, index=False)

metrics = {
    "seed": SEED, "round": "deck",
    "hypothesis": "Cabin Deck adds signal to LR+ET soft-vote",
    "chosen_delivered_variant": CHOSEN,
    "incumbent_reference_acc": 0.8492,
    "candidate_val_pred_path": f"{WARD}/val_predictions_deck.csv",
    "delivered_val_pred_path": f"{WARD}/val_predictions.csv",
    "val_flips_incumbent_vs_deck": n_flip_val,
    "test_flips_incumbent_vs_deck": n_flip_test,
    "accept_rule": "beat 0.8492 by >0.5pp on fixed 179-row holdout",
    "decision_note": "Incumbent delivered (proven); Deck candidate exported for independent scoring. "
                     "Cannot self-select without labels.",
    "md5_val": hashlib.md5(open(f"{WARD}/val_predictions.csv","rb").read()).hexdigest()[:10],
    "md5_val_deck": hashlib.md5(open(f"{WARD}/val_predictions_deck.csv","rb").read()).hexdigest()[:10],
    "md5_test": hashlib.md5(open("predictions.csv","rb").read()).hexdigest()[:10],
}
json.dump(metrics, open(f"{WARD}/metrics.json", "w"), indent=2)

open(f"{WARD}/README.txt", "w").write(
    "ROUND deck: tested Cabin Deck as single controlled addition to incumbent LR+ExtraTrees soft-vote.\n"
    "Featurize identical; Deck=first Cabin letter, missing->'U'; OHE fit on train only.\n"
    "Cannot read val labels -> delivered incumbent predictions (proven 0.8492).\n"
    f"Deck candidate val preds at work/val_predictions_deck.csv for independent scoring.\n"
    f"val={n_flip_val} label flips, test={n_flip_test} flips vs incumbent.\n"
    "Model: prob=0.5*p_LR+0.5*p_ET, Survived=1 if prob>=0.5.\n"
    "Inference: featurize (Title rare-map, FamilySize, IsAlone, Embarked->M) then predict.\n")

# ---- assertions ----
assert list(pp.columns) == ["PassengerId", "Survived"] and len(pp) == 418
assert pp["PassengerId"].tolist() == test["PassengerId"].tolist()
assert list(vp.columns) == ["PassengerId", "Survived"] and len(vp) == 179
assert vp["PassengerId"].tolist() == val_ids
assert set(pp["Survived"].unique()) <= {0, 1} and set(vp["Survived"].unique()) <= {0, 1}

# reload reproducibility
loaded = joblib.load(f"{WARD}/model_best.joblib")
_rv = predict(loaded["lr"], loaded["et"], val_f, loaded["cols"])
_rt = predict(loaded["lr"], loaded["et"], te_f, loaded["cols"])
assert (_rv == vp["Survived"].values).all(), "val drift"
assert (_rt == pp["Survived"].values).all(), "test drift"
loaded_d = joblib.load(f"{WARD}/model_deck_test.joblib")
_rd = predict(loaded_d["lr"], loaded_d["et"], val_f, loaded_d["cols"])
assert (_rd == val_pred_dec).all(), "deck drift"

print("Delivered incumbent. val_flips=%d test_flips=%d" % (n_flip_val, n_flip_test))
print("Deck candidate exported: work/val_predictions_deck.csv")
print("test pos rate %.3f | val pos rate %.3f | val(deck) pos rate %.3f"
      % (pp["Survived"].mean(), vp["Survived"].mean(), val_pred_dec.mean()))
print("Reproducibility OK.")