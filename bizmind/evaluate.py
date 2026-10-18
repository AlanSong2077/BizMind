"""Build a fixed local validation split and score experiment predictions."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import threading

from bizmind.config import Config
from bizmind.utils import generate, Conversation, get_logger
from bizmind.utils.execute import execute, execute_file


@dataclass
class EvaluationResult:
    score: float | None
    success: bool
    message: str


class Evaluator:
    def __init__(self, cfg: Config, is_lower_better: bool):
        self.cfg = cfg
        self.is_lower_better = is_lower_better
        self.workspace = (cfg.agent_workspace_dir / "evaluation").resolve()
        self.public_dir = self.workspace / "public"
        self.private_dir = self.workspace / "private"
        self.public_dir.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.sample_prediction_path = self.public_dir / f"validation_sample_{cfg.agent_prediction_file_name}"
        self.split_dataset_script_path = self.workspace / "split_dataset.py"
        self.eval_script_path = self.workspace / "evaluate.py"
        self.logger = get_logger("Evaluator", self.workspace / "evaluate.log")
        self.llm_logger = get_logger("LLM (Evaluator)", self.workspace / "llm.log", file_only=True)
        self.llm = Conversation(cfg.llm, self.llm_logger)
        # Generated scoring scripts share a report path. Serialize writes and reads together.
        self._evaluation_lock = threading.Lock()

    def _read_report(self) -> EvaluationResult:
        try:
            report = json.loads((self.private_dir / "eval_report.json").read_text(encoding="utf-8"))
            if report.get("success") is not True:
                return EvaluationResult(None, False, str(report.get("message", "Evaluation failed.")))
            score = report.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                return EvaluationResult(None, False, "Evaluator must return a finite numeric validation score.")
            return EvaluationResult(float(score), True, str(report.get("message", "")))
        except (OSError, ValueError, TypeError, AttributeError) as error:
            return EvaluationResult(None, False, f"Invalid evaluation report: {error}")

    def evaluate(self, prediction: Path) -> EvaluationResult:
        with self._evaluation_lock:
            (self.private_dir / "eval_report.json").unlink(missing_ok=True)
            result = execute_file(self.workspace, self.eval_script_path, [
                "--public_dir", self.public_dir, "--private_dir", self.private_dir,
                "--pred", prediction.resolve(),
            ])
            if not result.success:
                return EvaluationResult(None, False, f"Evaluation script failed: {result.error or result.output}")
            return self._read_report()

    def _get_instruction(self) -> str:
        return f"""
You are BizMind ML's independent local evaluation engineer.
Generate two self-contained Python 3 scripts for this business task:
<task_desc>{self.cfg.task_description}</task_desc>
<data_preview root="{self.cfg.task_input_dir.resolve()}">
{generate(self.cfg.task_input_dir)}
</data_preview>

The task defines the target, identifier columns, primary metric, metric direction,
output schema, constraints and validation strategy. Inspect actual input files.
Optional existing_solution/existing_code fields are background references only.
Use the current task's evaluation and output requirements to define the split
and scorer; historical scores or validation rules in reference code do not override them.
Use only supplied local data; do not download resources or fabricate missing data.
The source data is read-only. Scripts must run on CPU with the current interpreter.

1. split_dataset.py
Accept --input_dir, --public_dir and --private_dir. Follow task-specific time/group
split rules and prevent leakage. If no strategy is given, use a reproducible 80/20
split (seed 42), stratified only when appropriate for classification.
Keep validation labels only in --private_dir. Store labeled training rows,
unlabeled validation inputs and the unchanged unlabeled prediction set in
--public_dir. Keep the runtime dataset mapping's train_data/ and predict_data/
relative paths. Replace the training source with its training split; never link
or copy the full original labeled source into the public directory. For directory
inputs, remove validation labels from every associated table or metadata file.
Put validation inputs in validation_data/ and document their schema. Generate
{self.sample_prediction_path.name} with valid row IDs and the required prediction
columns so the evaluation script can be smoke-tested. Print the resulting layout,
row counts and split strategy. Extract archives only in the evaluation workspace.

2. evaluate.py
Accept --public_dir, --private_dir and --pred. Score the supplied validation
predictions against private ground truth using the task's exact primary metric.
Validate row IDs, row counts, duplicate IDs, columns, probability ranges and finite
values as applicable. Align rows by IDs when available, otherwise require the
original order. Never train a model or change the split during evaluation.
On invalid input return a descriptive failure rather than an invented score.
Print and write --private_dir/eval_report.json with exactly this schema:
{{"score": 0.5, "success": true, "message": ""}}
On failure use score: null, success: false, and explain the reason in message.

We run split_dataset.py first, then evaluate.py against the sample prediction file.
Use relative public/private paths supplied on the command line. Always provide full
script content. Respond with:
<current_file>split_dataset.py or evaluate.py; None only after both work</current_file>
<explanation>Implementation and validation reasoning</explanation>
<code>Full Python script, without Markdown fences; None when finished</code>
"""

    def setup(self):
        self.llm.add_message(role="system", content=self._get_instruction())
        split_succeeded = False
        for _ in range(self.cfg.agent_num_iterations_code_agents):
            def check(response):
                return response["current_file"][0] in {"split_dataset.py", "evaluate.py", "None", "none"}
            response = self.llm.query(required_fields=["current_file", "code"], check_fn=check)
            filename = response["current_file"][0]
            if filename.lower() == "none":
                self.llm.add_message(role="user", content="Both scripts must execute and produce a valid sample score before finishing.")
                continue
            args = ["--public_dir", self.public_dir, "--private_dir", self.private_dir]
            if filename == "split_dataset.py":
                split_succeeded = False
                for directory in (self.public_dir, self.private_dir):
                    shutil.rmtree(directory)
                    directory.mkdir(parents=True)
                args += ["--input_dir", self.cfg.task_input_dir.resolve()]
            else:
                (self.private_dir / "eval_report.json").unlink(missing_ok=True)
                args += ["--pred", self.sample_prediction_path]
            result = execute(self.workspace, response["code"][0], filename, args)
            feedback = f"Execution success: {result.success}\nOutput:\n{result.output[-12000:]}\nError: {result.error or ''}"
            if filename == "split_dataset.py":
                split_succeeded = result.success and self.sample_prediction_path.is_file()
            if filename == "evaluate.py" and result.success:
                report = self._read_report()
                feedback += f"\nSample scoring: {report}"
                if split_succeeded and report.success and self.split_dataset_script_path.is_file():
                    return
            self.llm.add_message(role="user", content=feedback + "\nCorrect failures or generate the next script. Use the same current_file/explanation/code tags. Return None only after both scripts work.")
        raise RuntimeError("Local evaluator setup exceeded the configured code-iteration budget.")
