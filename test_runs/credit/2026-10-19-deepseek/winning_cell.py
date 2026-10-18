import os, json, numpy as np, pandas as pd
import lightgbm as lgb

SEED = 42
os.environ.setdefault("OMP_NUM_THREADS", "4")
BASE = "../input/data"
train = pd.read_csv(f"{BASE}/train_data/train.csv")
test  = pd.read_csv(f"{BASE}/predict_data/test.csv")
val   = pd.read_csv(f"{BASE}/validation_data/validation_features.csv")

delinq = ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes90DaysLate","NumberOfTime60-89DaysPastDueNotWorse"]
impute_cols = ["MonthlyIncome","NumberOfDependents"]
medians = {c: float(train[c].median()) for c in impute_cols}

def build(df):
    X = df.copy()
    X["MonthlyIncome_missing"] = X["MonthlyIncome"].isna().astype(int)
    X["NumberOfDependents_missing"] = X["NumberOfDependents"].isna().astype(int)
    X["delinq_sentinel"] = X[delinq].isin([96,98]).any(axis=1).astype(int)
    for c in delinq:
        X[c] = X[c].where(~X[c].isin([96,98]), 20)
    ru = X["RevolvingUtilizationOfUnsecuredLines"]
    X["ru_clipped"] = ru.clip(0, 1.5)
    X["ru_gt1_flag"] = (ru > 1).astype(int)
    X["DebtRatio_is_income"] = (X["DebtRatio"] > 10).astype(int)
    for c, m in medians.items():
        X[c] = X[c].fillna(m)
    X["total_delinq"] = X[delinq[0]] + X[delinq[1]] + X[delinq[2]]
    X["income_per_dependent"] = X["MonthlyIncome"] / (X["NumberOfDependents"] + 1.0)
    X["debt_per_income"] = X["DebtRatio"] / (X["MonthlyIncome"] + 1.0)
    X["log_MonthlyIncome"] = np.log1p(X["MonthlyIncome"].clip(lower=0))
    X["log_DebtRatio"] = np.log1p(X["DebtRatio"].clip(lower=0))
    X["delinq_weighted"] = (X[delinq[1]]*3 + X[delinq[2]]*2 + X[delinq[0]])
    X["ru_x_delinq"] = X["ru_clipped"] * (1 + X["total_delinq"])
    X["debt_per_line"] = X["DebtRatio"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1.0)
    X["income_per_line"] = X["MonthlyIncome"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1.0)
    X["has_delinq_any"] = (X["total_delinq"] > 0).astype(int)
    X["age_x_delinq"] = X["age"] * (1 + X["has_delinq_any"])
    X["re_loan_ratio"] = X["NumberRealEstateLoansOrLines"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1.0)
    X["income_per_person"] = X["MonthlyIncome"] / (X["NumberOfDependents"] + 1.0)
    X["util_x_dti"] = X["ru_clipped"] * X["DebtRatio"]
    X["debt_pressure"] = X["DebtRatio"] * X["MonthlyIncome"] / 1e3
    X["ru_over_lines"] = X["ru_clipped"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1.0)
    X["sev_delinq_rate"] = X["NumberOfTimes90DaysLate"] / (X["total_delinq"] + 1.0)
    return X

feat = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents',
        'MonthlyIncome_missing','NumberOfDependents_missing','delinq_sentinel','ru_clipped','ru_gt1_flag',
        'DebtRatio_is_income','total_delinq','income_per_dependent','debt_per_income',
        'log_MonthlyIncome','log_DebtRatio','delinq_weighted','ru_x_delinq','debt_per_line',
        'income_per_line','has_delinq_any','age_x_delinq','re_loan_ratio',
        'income_per_person','util_x_dti','debt_pressure','ru_over_lines','sev_delinq_rate']

full = build(train)[feat]; te = build(test)[feat]; va = build(val)[feat]
y = train["SeriousDlqin2yrs"].values
n_it = 371

def make(seed, n_it):
    return lgb.LGBMClassifier(objective="binary", num_leaves=15, learning_rate=0.03,
                              n_estimators=n_it, min_child_samples=40, subsample=0.9,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=4, metric="auc", verbose=-1)

seeds = [42,7,123,2024,555,13,99,314,101,777,222,999,1,2,3,11,17,23,29,37,41,53,67,79]
vp=np.zeros(len(va)); tp=np.zeros(len(te))
for sd in seeds:
    m=make(sd,n_it); m.fit(full,y)
    vp+=m.predict_proba(va)[:,1]; tp+=m.predict_proba(te)[:,1]
vp/=len(seeds); tp/=len(seeds)
vp=np.clip(vp,0,1); tp=np.clip(tp,0,1)
assert np.isfinite(vp).all() and np.isfinite(tp).all()

os.makedirs("artifacts", exist_ok=True)
vout = pd.DataFrame({"Id":val["Id"].values,"Probability":vp})
vout.to_csv("validation_predictions.csv", index=False)
po = pd.DataFrame({"Id":test["Id"].values,"Probability":tp})
assert len(po)==101503 and list(po.columns)==["Id","Probability"] and po["Id"].is_unique
po.to_csv("predictions.csv", index=False)
assert len(vout)==30000 and vout["Id"].is_unique and set(vout["Id"])==set(val["Id"])
assert list(po["Id"].values)==list(test["Id"].values)

json.dump({"n_it":n_it,"seeds":seeds,"features":feat,"medians":medians},
          open("artifacts/run_meta_bag33_s24.json","w"), indent=2)
with open("artifacts/README.md","w") as f:
    f.write(f"Champion config: build() 33 features (train medians impute, delinq sentinel cap, RU clip). "
            f"LGBM num_leaves=15 lr=0.03 n_it={n_it} subsample=0.9 colsample=0.8 reg_lambda=1.0 "
            f"min_child_samples=40. {len(seeds)}-seed bag {seeds} averaged predict_proba[:,1].\n")
print("done rows:", len(vout), len(po))
