# Example datasets

Ready-to-use CSVs are included so both example configurations can run directly.

| Directory | Source | Included inputs |
| --- | --- | --- |
| `titanic/` | [Kaggle Titanic](https://www.kaggle.com/c/titanic) | 891 training rows, 418 prediction rows and sample output |
| `give_me_some_credit/` | [Kaggle public dataset](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset), from [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) | 150,000 training rows, 101,503 prediction rows, sample output and original data dictionary |

`sources.json` records the downloaded source files and their SHA-256 checksums.
For the credit dataset, `prepared_data.json` records the included input checksums:
preparation renamed the unnamed identifier to `Id` and removed the entirely empty
test target column. Feature values, missing values and row order were preserved.
Training and test IDs are numbered separately in this dataset and may overlap.

Sample submission files describe the output format; they are not test labels.
BizMind stages only `train.csv` and `test.csv` for each example. Download archives,
duplicate original credit CSVs and raw page snapshots remain excluded from Git.

Dataset attribution and usage terms are provided on the linked Kaggle pages.
