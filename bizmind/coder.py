"""Implement and refine a local experiment using executable validation evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING

from bizmind.config import Config
from bizmind.experiments import Draft, Pipeline
from bizmind.evaluate import Evaluator
from bizmind.utils import Conversation, generate, WorstMetricValue, MetricValue, get_logger
from bizmind.utils.jupyter_session import JupyterSession, ExecutionResult

if TYPE_CHECKING:
    from bizmind.agent import MetricUpdater


class CodeAgent:
    def __init__(self, cfg: Config, draft: Draft, is_lower_better: bool,
                 metric_updater: "MetricUpdater", evaluator: Evaluator, env_name: str):
        self.cfg = cfg
        self.draft = draft
        self.is_lower_better = is_lower_better
        self.metric_updater = metric_updater
        self.evaluator = evaluator
        self.prediction_name = cfg.agent_prediction_file_name
        self.start_time = time.time()
        self.best_code = None
        self.best_metric = WorstMetricValue()
        self.last_feedback = "No code cells have executed yet."
        self.logger = get_logger(f"coder-{draft.id}", cfg.agent_workspace_dir / "coder.log")
        self.llm = Conversation(cfg.llm, logger=get_logger(
            f"llm-{draft.id}", cfg.agent_workspace_dir / "llm.log", file_only=True))
        (cfg.agent_workspace_dir / "best_prediction").mkdir(parents=True, exist_ok=True)
        self.best_prediction_path = cfg.agent_workspace_dir / "best_prediction" / self.prediction_name
        self.jupyter_session = JupyterSession(cfg, env_name)

    def _response_format(self) -> str:
        return """
Respond with exactly one executable notebook cell, using these tags:
<goal>Hypothesis, expected runtime, expected output and how it tests the hypothesis</goal>
<code>Python code for the next cell, without Markdown fences</code>
<validation_prediction_file>Relative file path for complete validation predictions, or None</validation_prediction_file>
<prediction_file>Relative file path for complete business prediction output, or None</prediction_file>
Both file tags must be None unless this cell writes BOTH complete outputs.
Do not report an estimated score: the independent local evaluator measures it.
"""

    def _post_initial_message(self):
        prompt = f"""
You are BizMind ML's implementation engineer for a local business ML experiment.
<task_desc>{self.cfg.task_description}</task_desc>
<experiment>{self.draft.description}</experiment>
<code_outline>{self.draft.code}</code_outline>

Read local data from ../input/data/. It contains the fixed public training split,
unlabeled validation inputs in validation_data/, a sample validation prediction
file and the unlabeled business prediction data. Inspect their actual schemas.
<data_preview>{generate(self.cfg.agent_workspace_dir.parent, include_file_details=False)}</data_preview>

The business task defines the target, primary metric, validation strategy, output
columns, row identifiers and constraints. Use only supplied local data and parent
artifacts, plus optional existing_solution/existing_code reference text in the task.
Use supplied references as a starting point: adapt data paths, target, features,
dependencies and outputs to the current requirements, then train and measure a
fresh baseline before improving it. Reference files and their neighboring modules
are not automatically copied into this workspace; supplied code is context, not
an executable parent checkpoint. Recreate needed code in the current workspace.
Treat historical scores as unverified and report which reference parts you reused
or changed. Empty reference fields mean no prior approach was supplied.
Do not download datasets, pretrained weights or predictions. Fit all
preprocessing on training rows only; never read hidden validation labels, source
training files outside ../input/data/, or the private evaluator workspace.
The validation split is fixed for all experiments; do not change it to improve scores.
You may create an internal training holdout for tuning without changing the fixed
external validation inputs. Record seeds and avoid target leakage.

Start with a working baseline, then change a small number of factors at a time.
Use observed execution errors and measured validation scores to choose improvements.
Explain gains and regressions; do not claim results that were not executed.
Write complete prediction outputs for both validation and the business prediction
set in the same cell. The business output must be ./{self.prediction_name} and
must preserve the task's identifiers, ordering and schema. Save model, preprocessing
and inference instructions in the working directory as requested by the task.
Parent artifacts, when supplied, are copied into ./ and can be reused; outputs
from other workers in the current round are unavailable. Check alignment before
combining any predictions from your own local experiments.
Parent files may include later trials: check checkpoint metadata against the
parent's scored code before reuse; retrain if their provenance is uncertain.

A persistent Jupyter session is used. Propose one cell at a time, inspect its output,
repair failures, and then refine the best measured method within the budget.
Check package availability before use; use packages installed in this environment.
<execution_environment>
{self.jupyter_session.hardware.to_prompt()}
CPU core limit: {self.cfg.execution_max_cpu_cores}; keep n_jobs and thread counts within it.
Maximum time per code cell: {self.cfg.execution_timeout} seconds.
</execution_environment>
Use torch.device(os.environ['BIZMIND_DEVICE']) for PyTorch models and tensors.
Do not hardcode .cuda(). On MPS/CPU do not require CUDA-only operations or packages.
Use CPU for libraries without support for the detected accelerator and choose
model and batch sizes that fit available memory.
"""
        if self.draft.codebase_content:
            prompt += f"\n<completed_local_parent_code>{self.draft.codebase_content}</completed_local_parent_code>\n"
        self.llm.add_message(role="system", content=prompt + self._response_format())

    def _get_feedback_prompt(self, result: ExecutionResult) -> str:
        output = result.output or ""
        if len(output) > 12000:
            output = output[:6000] + "\n...\n" + output[-6000:]
        return (f"Execution time: {result.execution_time:.2f} seconds; success: {result.success}.\n"
                f"<output>{output}</output>\n<error>{result.error or ''}</error>\n"
                "Use this evidence to repair failures or test the next controlled improvement.\n")

    def _prediction_path(self, value: str) -> Path:
        path = Path(value)
        root = self.cfg.agent_workspace_dir.resolve()
        candidate = (root / path).resolve()
        if path.is_absolute() or not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"Prediction must be an existing file inside the working directory: {value}")
        return candidate

    def _evaluate_predictions(self, response: dict) -> str:
        if response["prediction_file"][0].strip().lower() == "none":
            return ""
        try:
            prediction = self._prediction_path(response["prediction_file"][0])
            validation = self._prediction_path(response["validation_prediction_file"][0])
            if prediction == validation:
                raise ValueError("Business and validation predictions must be separate files.")
            result = self.evaluator.evaluate(validation)
            if not result.success:
                return f"\nValidation failed: {result.message}\n"
            score = MetricValue(result.score, maximize=not self.is_lower_better)
            if score > self.best_metric:
                self.best_metric = score
                self.best_code = deepcopy(self.jupyter_session.cells)
                shutil.copyfile(prediction, self.best_prediction_path)
                self.metric_updater.post(score, self.best_prediction_path)
            return f"\nMeasured fixed-split validation metric: {score}. Best so far: {self.best_metric}.\n"
        except (OSError, ValueError) as error:
            return f"\nPrediction evaluation failed: {error}\n"

    def _generate_report(self) -> Pipeline:
        full_code = self.best_code.get_notebook_code() if self.best_code is not None else ""
        if full_code:
            (self.cfg.agent_workspace_dir / "best_code.py").write_text(full_code, encoding="utf-8")
        self.llm.add_message(role="user", content=f"""
Summarize this local experiment using executed evidence only.
Engine-recorded best validation metric: {self.best_metric}.
Latest execution feedback: {self.last_feedback}
<best_code>{full_code}</best_code>
Describe the best measured version, not an unexecuted or later regressed version.
If no valid score exists, explain failure and do not claim success.
Respond with:
<description>Hypothesis, data processing, parameters, validation evidence, changes
from the parent, artifact paths and how to load the saved model</description>
<code>Concise outline of the best measured code, or failed approach</code>
<suggestions>Observed weaknesses, failures, lessons and prioritized next experiments;
distinguish evidence from hypotheses</suggestions>
""")
        report = self.llm.query(required_fields=["description", "code", "suggestions"])
        prediction = self.best_prediction_path if not self.best_metric.is_worst else None
        (self.cfg.agent_workspace_dir / "experiment.json").write_text(json.dumps({
            "id": self.draft.id, "parent": self.draft.codebase, "score": self.best_metric.value,
            "description": report["description"][0], "suggestions": report["suggestions"][0],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return Pipeline(id=self.draft.id, title=self.draft.title, description=report["description"][0],
                        code=report["code"][0], full_code=full_code, metric=self.best_metric,
                        prediction=prediction, output_dir=self.cfg.agent_workspace_dir,
                        suggestions=report["suggestions"][0])

    def run(self) -> Pipeline:
        try:
            self._post_initial_message()
            for iteration in range(self.cfg.agent_num_iterations_code_agents):
                if time.time() - self.start_time >= self.cfg.agent_time_limit_code_agents:
                    break
                def check(response):
                    return ((response["prediction_file"][0].strip().lower() == "none")
                            == (response["validation_prediction_file"][0].strip().lower() == "none"))
                response = self.llm.query(required_fields=["code", "goal", "validation_prediction_file", "prediction_file"],
                                          check_fn=check)
                self.logger.info(f"Code iteration {iteration + 1}: {response['goal'][0]}\n{response['code'][0]}")
                result = self.jupyter_session.append_cell(response["code"][0], response["goal"][0])
                feedback = self._get_feedback_prompt(result)
                if result.success:
                    feedback += self._evaluate_predictions(response)
                self.last_feedback = feedback
                self.logger.info(feedback)
                self.llm.add_message(role="user", content=feedback + self._response_format())
            if self.best_prediction_path.is_file() and not self.best_metric.is_worst:
                shutil.copyfile(self.best_prediction_path, self.cfg.agent_workspace_dir / self.prediction_name)
            return self._generate_report()
        finally:
            self.jupyter_session.shutdown()
