# Give Me Some Credit — 2026-10-19

Two DeepSeek Flash-driven CPU experiment rounds completed: 16 code executions,
15 successful validation scores, and no failed rounds. One syntax error was
repaired in the following cell. The runtime used source commit `a7ab73f`; the
task and configuration snapshots are included in `config/`.

The evaluator used a fixed stratified 80/20 split with seed 42: **120,000 fitting
rows and 30,000 validation rows**. The best engine-recorded ROC-AUC was
**0.8705731652294517**. The prediction snapshot contains 101,503 rows with columns
`Id,Probability`. Official test labels were unavailable.

## Method changes

| Change | Fixed-validation ROC-AUC, rounded |
| --- | ---: |
| Baseline LightGBM with median imputation | 0.8680 |
| Missing indicators, delinquency handling and credit features | 0.8685 |
| Blend with logistic regression | 0.8693 |
| Regularized LightGBM and multi-seed averaging | 0.8700 |
| Expanded feature sets and further averaging | 0.8705 |
| 33 features, 24-seed LightGBM probability average | **0.870573** |

The winning configuration used 371 trees per seed, 15 leaves, learning rate 0.03
and four CPU threads. It first achieved the best score in the second round's sixth
code cell. The final shallow-tree trial scored 0.8705517150150165 and did not
replace the best prediction snapshot. Later gains from wider seed averaging were
small; these measurements do not establish a gain on a new independent holdout.

## Files

- `summary.json`, `best.json`, `experiments.json`: measured results and round history.
- `predictions.csv`: the best test prediction snapshot.
- `winning_cell.py`: the actual executed winning cell extracted from the worker log.
- `round-*/executed_cells.json`: each worker's eight original code cells.
- `round-*/coder.log`, `round-*/llm.log`: execution feedback and model conversations.
- `round-*/experiment.json`, `round-*/artifacts/`: generated reports and trial metadata.
- `evaluation/`: generated split/scoring scripts and evaluator conversation.
- `logs/planner_llm.log`, `agent.log`, `metric.log`: planning, rounds and score updates.
- `manifest.json`: checksums of archived files.

Worker logs retain inherited parent history; `executed_cells.json` isolates each
worker's own cells. Model-generated hypotheses and reports are preserved as
recorded. The score comes from the independent evaluator. The archived winning
cell has not been separately replayed. Absolute project/home paths are replaced
with placeholders, and credentials and kernel connection files are excluded.
The original logger stripped non-ASCII characters.

## Run

The prepared input CSVs are included in `data/give_me_some_credit/`. From the
project root, with your DeepSeek key in `.env`, start a new experiment with:

```sh
./start.sh --config examples/credit_training.yaml
```

See [the task description](../../../examples/credit_description.md) for data
sources, validation rules and output details.
