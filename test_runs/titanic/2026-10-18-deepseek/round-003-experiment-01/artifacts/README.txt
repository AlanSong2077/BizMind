FINAL DELIVERABLE: proven incumbent LR+ExtraTrees soft-vote (external acc 0.8492).
Config: w=0.5, ET n=400 max_depth=6, min_samples_leaf=4, threshold=0.5.
Model: prob=0.5*p_LR+0.5*p_ET; Survived=1 if prob>=0.5. Seed 42, CPU n_jobs=4.
Preprocess (fit on 712 train rows only): FamilySize, IsAlone, Title rare-map, Embarked NA->M;
  Num(median+scale), Age/Fare(median+indicator+scale), Cat(Sex,Embarked,Title)->OHE.
Inference: featurize -> ColumnTransformer -> blend proba -> >=0.5 -> int 0/1.
Deliverables: predictions.csv(418), work/val_predictions.csv(179), work/model_best.joblib.
Negative evidence: deck, add_GB, interactions, w40(0.8436), title_age, thr0.58(0.8268) all < 0.8492.
