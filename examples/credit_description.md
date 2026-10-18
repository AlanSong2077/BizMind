# Give Me Some Credit local test

A medium-size tabular binary classification task: predict the probability of
serious delinquency within the next two years. The target is `SeriousDlqin2yrs`;
the original competition metric is ROC-AUC, maximized. Output probabilities, not
thresholded class labels.

The downloaded Kaggle public dataset contains 150,000 labeled training rows and
101,503 prediction rows, with 10 numeric features. The positive class contains
10,026 training rows (6.684%). The features cover credit utilization, age, income,
debt ratio, past delinquencies, credit accounts, real-estate loans and dependents.

## Data on this machine

Ready-to-use files are included under `data/give_me_some_credit/`:

- `sampleEntry.csv`: sample output format, not ground truth.
- `Data Dictionary.xls`: downloaded original dictionary.
- `train.csv`, `test.csv`: prepared local inputs used by BizMind.
- `sources.json`, `prepared_data.json`: provenance, transformations and checksums.

The downloaded originals, archive and source-page snapshots remain local.

Preparation only renames `Unnamed: 0` to `Id` and removes the entirely empty target
column from the prediction data. Features, missing values and row order remain
unchanged. IDs are unique within each file, but overlap between training and test:
they must never be used to look up labels across datasets.

No download or preparation script is required. The prepared CSVs preserve all
150,000 training rows and 101,503 prediction rows.

## Run the experiment yourself

From the BizMind project root, with the existing `.env` containing your DeepSeek
key:

```sh
./start.sh --config examples/credit_training.yaml
```

The configuration runs two sequential CPU rounds, with at most eight code cells
per worker, four CPU threads, a 300-second cell timeout and a 1,800-second worker
budget. The evaluator has a separate eight-step setup budget. LLM calls and
planning/reporting have their own timeout, so these limits are not a total wall
clock guarantee.

The evaluator creates a fixed stratified split with seed 42: 120,000 fitting rows
and 30,000 validation rows. Coding workers receive the already-split training
data and validation features without labels. All rounds reuse the same validation
IDs. Preprocessing is fitted only on the fitting subset.

Results go to `workspace/credit-risk-local/<run-time>/`. Check `best.json` for the
measured ROC-AUC, `experiments.json` for round reports and
`predictions/predictions.csv` for the best output. The prediction file must have
exactly `Id,Probability`, 101,503 rows, and the original test ID order.

The completed two-round experiment reached **0.8705731652294517 local validation
ROC-AUC** using 33 engineered features and a 24-seed LightGBM ensemble. The model
used 371 trees per seed, 15 leaves and a learning rate of 0.03. See the archived
logs and results in `test_runs/credit/2026-10-19-deepseek/`.

Test labels are unavailable, so local validation ROC-AUC is not an official
competition score.

## Sources

- [Downloaded public Kaggle dataset](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
- [Original competition](https://www.kaggle.com/c/GiveMeSomeCredit)
- [Official evaluation](https://www.kaggle.com/c/GiveMeSomeCredit/overview/evaluation)

The download is a one-time preparation step; BizMind reads local files at runtime.
