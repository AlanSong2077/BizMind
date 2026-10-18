BizMind is an AI agent for business-driven machine learning that improves models through iterative local experiments and validation. This project is adapted based on Comind.

## Experiment results

Both experiments used DeepSeek Flash to plan and generate code, with model training
on CPU using four threads. A fixed stratified 80/20 split with seed 42 was reused
across all rounds. Scores below are local validation results.

| Task | Training / validation rows | Prediction rows | Rounds | Best method | Result |
| --- | ---: | ---: | ---: | --- | --- |
| [Titanic](examples/titanic_description.md) | 712 / 179 | 418 | 3 | Equal-weight LogisticRegression + ExtraTrees probabilities | **84.92% accuracy** (152/179) |
| [Give Me Some Credit](examples/credit_description.md) | 120,000 / 30,000 | 101,503 | 2 | LightGBM with 33 features and a 24-seed probability average | **0.870573 ROC-AUC** |

### Training models

- **Titanic:** LogisticRegression (L2, `C=1.0`) and ExtraTreesClassifier (400 trees,
  depth 6, minimum 4 samples per leaf). Average their probabilities with equal
  weights and classify at a 0.5 threshold.
- **Give Me Some Credit:** LightGBM with 371 trees, 15 leaves and learning rate
  0.03 per model. Average probabilities from 24 random seeds using 33 features.

### Iteration summary

Each round contains multiple code-and-evaluation steps on the same fixed split.

| Task / round | Executed changes and outcome | Best local result |
| --- | --- | ---: |
| Titanic / 1 | Built a logistic baseline, added name-derived titles and family features, tested tree models, and selected the LogisticRegression + ExtraTrees blend. | **84.92% accuracy** |
| Titanic / 2 | Tested blend weights, ExtraTrees parameters and feature interactions. Trials tied or regressed, so the first-round model was retained. | **84.92% accuracy** |
| Titanic / 3 | Tested cabin deck, title-based age imputation and a GradientBoosting branch. No improvement; retained the existing best model. | **84.92% accuracy** |
| Credit / 1 | Improved the approximately 0.8680-AUC baseline through missing/sentinel flags, ratio and interaction features, LightGBM tuning and blending trials. Selected 33 features with a five-seed LightGBM average. | **0.870504 ROC-AUC** |
| Credit / 2 | Compared 8-, 12- and 24-seed averages, tail transforms, tree variants and a logistic branch. The 24-seed average scored highest; the gain over round 1 was approximately 0.000069 AUC. | **0.870573 ROC-AUC** |

Lower-scoring trials did not overwrite the best prediction snapshots. Later-round
gains were small or absent, so these runs demonstrate iteration and model
selection without establishing a gain on a new independent holdout.

[Titanic logs and reproduction](test_runs/titanic/2026-10-18-deepseek/README.md) ·
[Credit experiment logs and results](test_runs/credit/2026-10-19-deepseek/README.md)

## Run locally

Task YAML files accept two optional inputs: `existing_solution` and `existing_code`.
Omit them or use `null` / an empty string to start without prior references. Both
accept multiline text or a UTF-8 text file via `path`, resolved relative to the
task YAML. For example:

```yaml
existing_solution: |
  Use median imputation and LightGBM; improve recall under the new requirements.
existing_code:
  path: ./baseline.py
```

To paste code directly, use `existing_code: |` followed by indented code. Reference
text reaches the planner and coding agent; it is not executed during input loading
and neighboring files are not copied automatically. The agent adapts the reference
to the current task and measures a fresh baseline before iterating.

Training and prediction CSVs are included in [data/](data/README.md). Install the
dependencies, put your `DEEPSEEK_API_KEY` in `.env`, then run from the project root:

```sh
./start.sh --config examples/local_training.yaml   # Titanic
./start.sh --config examples/credit_training.yaml  # Give Me Some Credit
```

Each run writes its scores, experiment history and best predictions under
`workspace/<task-id>/<run-time>/`.
