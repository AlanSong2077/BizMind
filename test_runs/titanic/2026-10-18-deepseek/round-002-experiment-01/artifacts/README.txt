FINAL - best-measured incumbent LR+ExtraTrees soft-vote (fixed-val 0.8492)
Config: w=0.5, LR C=1.0, ET n=400 depth=6 leaf=4 criterion=gini, threshold=0.5.
Features: num[Pclass,SibSp,Parch,FamilySize,IsAlone]; af[Age(median+indicator,scaled),Fare]; cat[Sex,Embarked(missing->M),Title (Mlle/Ms->Miss,Mme->Mrs,rare<10->Rare)].
Inference: prob = 0.5*p_LR + 0.5*p_ET; Survived=1 if prob>=0.5 else 0.
Fit on all 712 train_data/train.csv rows; validation/test transform-only. Seed 42, CPU, n_jobs=4.
Perturbations rejected (none beat 0.8492): r4 CV=0.8436, r6 interactions=0.8212, r8 GB=0.8436, r10 more trees=0.8436, r12 thr0.45=0.8380, r16 seed-bag=0.8436, r18 LR C=0.3=0.8380; r21 entropy criterion produced identical predictions.
Pinned outputs: val md5=cceba64816, test md5=42c718fef6.
Deliverables: predictions.csv (418), work/val_predictions.csv (179) from same model.
