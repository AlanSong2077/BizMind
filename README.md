BizMind is an AI agent for business-driven machine learning that improves models through iterative local experiments and validation. This project is adapted based on Comind.

## Experiment results

Both experiments used DeepSeek Flash to plan and generate code, with model training
on CPU using four threads. A fixed stratified 80/20 split with seed 42 was reused
across all rounds. Scores below are local validation results.

| Task | Training / validation rows | Prediction rows | Rounds | Best method | Result |
| --- | ---: | ---: | ---: | --- | --- |
| Titanic | 712 / 179 | 418 | 3 | Equal-weight LogisticRegression + ExtraTrees probabilities | **0.8492 accuracy** |
| Give Me Some Credit| 120,000 / 30,000 | 101,503 | 2 | LightGBM with 33 features and a 24-seed probability average | **0.8705 ROC-AUC** |

## Summary

Agent finished a good model selection task before running code, and agents contains multiple code-and-evaluation steps on the same fixed split while running.

- **Titanic:** LLM chooses LogisticRegression and ExtraTreesClassifier to optimize.
- **Give Me Some Credit:** LLM chooses LightGBM with 371 trees, 15 leaves and learning rate 0.03 per model.

Finally, Logs were writed in test files, recording all the improvements, metrics, exceptions, LLM coding error during loop.

## Run

Training and prediction CSVs are included in [data/](data/README.md). Install the dependencies, put your `DEEPSEEK_API_KEY` in `.env`, then run from the project root. 
```sh
./start.sh --config examples/local_training.yaml  
```
You can build your task including task description, overview, trainning data, testing data,  init code to begin a new work.
