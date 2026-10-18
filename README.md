BizMind is an AI agent for business-driven machine learning that improves models through iterative local experiments and validation. This project is adapted based on Comind.

## Experiment results

Both experiments used DeepSeek Flash to plan and generate code, with model training
on CPU using four threads. A fixed stratified 80/20 split with seed 42 was reused
across all rounds. Scores below are local validation results.

| Task | Training / validation rows | Prediction rows | Rounds | Best method | Result |
| --- | ---: | ---: | ---: | --- | --- |
| [Titanic](examples/titanic_description.md) | 712 / 179 | 418 | 3 | Equal-weight LogisticRegression + ExtraTrees probabilities | **84.92% accuracy** (152/179) |
| [Give Me Some Credit](examples/credit_description.md) | 120,000 / 30,000 | 101,503 | 2 | LightGBM with 33 features and a 24-seed probability average | **0.870573 ROC-AUC** |

Titanic improved through name-derived titles, family features and model blending.
The credit experiment improved from approximately 0.8680 AUC through missing-value
and delinquency features, LightGBM tuning and multi-seed averaging. Lower-scoring
trials did not overwrite the best prediction snapshots.

[Titanic logs and reproduction](test_runs/titanic/2026-10-18-deepseek/README.md) ·
[Credit experiment logs and results](test_runs/credit/2026-10-19-deepseek/README.md)

## Run locally

Training and prediction CSVs are included in [data/](data/README.md). Install the
dependencies, put your `DEEPSEEK_API_KEY` in `.env`, then run from the project root:

```sh
./start.sh --config examples/local_training.yaml   # Titanic
./start.sh --config examples/credit_training.yaml  # Give Me Some Credit
```

Each run writes its scores, experiment history and best predictions under
`workspace/<task-id>/<run-time>/`.
