# Titanic local test case

Predict each passenger's survival from passenger attributes. The target is
`Survived`: `0` means deceased and `1` means survived. The official metric is
**accuracy**: correct predictions divided by the number of passengers, maximized.

## Local files

The official competition archive was downloaded into `data/titanic/`:

| File | Rows | Purpose |
| --- | ---: | --- |
| `train.csv` | 891 | Features and known survival labels |
| `test.csv` | 418 | Features for inference; no survival labels |
| `gender_submission.csv` | 418 | Example predictions, **not ground truth** |

`data/titanic/sources.json` records the download time and SHA-256 checksums.
Official task text (`description.md`), evaluation (`evaluation.html`), and data
dictionary (`data_description.html`) are archived under `data/titanic/source/`,
alongside the raw page HTML, competition metadata, and page API response.
Downloaded files and generated results stay local and are ignored by Git.
On another machine, download the official competition ZIP after accepting its
rules and extract these CSV files into `data/titanic/`.

## Fields

| Column | Meaning |
| --- | --- |
| `PassengerId` | Passenger identifier used to align predictions |
| `Survived` | Binary survival label, present only in training data |
| `Pclass` | Ticket class, from first (`1`) to third (`3`) |
| `Name` | Passenger name |
| `Sex` | Passenger sex |
| `Age` | Age in years, including fractional ages for infants |
| `SibSp` | Number of siblings and spouses aboard |
| `Parch` | Number of parents and children aboard |
| `Ticket` | Ticket identifier |
| `Fare` | Passenger fare |
| `Cabin` | Cabin identifier |
| `Embarked` | Embarkation port: C = Cherbourg, Q = Queenstown, S = Southampton |

Handle missing values using transformations fitted only on the training subset.

## Local validation and output

Use a single stratified 80/20 split with random seed 42: 712 training passengers
and 179 validation passengers. Reuse that split across all experiment rounds.
This holdout is our local testing choice; its score is not an official leaderboard
score. Since the 418 test labels are hidden, no local test-set accuracy is known.

`predictions.csv` must have exactly `PassengerId,Survived`, with 418 rows, binary
integer predictions, and no DataFrame index. This fixture additionally preserves
the original test row order. Keep validation predictions separate for local scoring.

## Run

From the BizMind project root, check the data and train a deterministic CPU baseline
without any model API calls:

```sh
.venv/bin/python examples/titanic_baseline.py
```

The check writes predictions, a fitted preprocessing/model pipeline, and a measured
local validation report into `workspace/titanic-baseline/`. It verifies the official
CSV checksums, schemas, identifiers, and prediction format. It uses logistic
regression with numeric imputation/scaling and categorical imputation/one-hot
encoding. Both validation and test predictions use the same model fitted on the
712-row training subset.

For the actual DeepSeek-driven self-iteration, set `DEEPSEEK_API_KEY` in your
terminal, then run:

```sh
./start.sh --config examples/local_training.yaml
```

The configuration runs three sequential CPU experiment rounds. The evaluator
creates the fixed split and scores model predictions. Only the two input CSVs are
staged into the run; the sample submission and downloaded metadata are excluded.
Results are written under `workspace/titanic-local/`.

## Official sources

- [Competition task and evaluation](https://www.kaggle.com/c/titanic)
- [Data description and dictionary](https://www.kaggle.com/c/titanic/data)
- [Official competition archive](https://www.kaggle.com/api/v1/competitions/data/download-all/titanic)

The task and metric above are summarized from these sources. This fixture uses
downloaded local files; BizMind does not require a Kaggle runtime integration.
