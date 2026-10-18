"""Plan, execute, and learn exclusively from local business experiments."""
import hashlib
import json
import queue
import threading
import time
from copy import deepcopy
from pathlib import Path

from rich.status import Status

from bizmind.config import Config
from bizmind.experiments import Draft, Pipeline
from bizmind.strategy import iteration_focus
from bizmind.utils import query_llm, MetricValue, WorstMetricValue, extract_fields, copytree, get_logger
from bizmind.utils.hardware import detect_hardware
from bizmind.evaluate import Evaluator


class MetricUpdater:
    """Consume every result before a round ends; snapshot files at posting time."""
    def __init__(self, cfg: Config, best_metric: MetricValue, agent=None):
        self.cfg = cfg
        self.q = queue.Queue()
        self.best_metric = best_metric
        self.start_time = cfg.start_time
        self.logger = get_logger("MetricUpdater", cfg.agent_workspace_dir / "metric.log")

    def post(self, metric: MetricValue, prediction: Path):
        if not isinstance(metric, WorstMetricValue):
            self.q.put((metric, prediction.read_bytes()))

    def stop(self):
        self.q.put(None)

    def _update_best_metric(self, metric: MetricValue, prediction: bytes):
        if metric > self.best_metric:
            self.best_metric = metric
            elapsed = time.time() - self.start_time
            output = self.cfg.agent_workspace_dir / "predictions"
            (output / f"{elapsed:.6f}_{self.cfg.agent_prediction_file_name}").write_bytes(prediction)
            (output / self.cfg.agent_prediction_file_name).write_bytes(prediction)
            (self.cfg.agent_workspace_dir / "best.json").write_text(json.dumps({
                "score": metric.value, "maximize": metric.maximize,
                "prediction_file": str(output / self.cfg.agent_prediction_file_name),
            }, indent=2), encoding="utf-8")
            self.logger.info(f"Global best validation metric: {metric}")

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            self._update_best_metric(*item)


from bizmind.coder import CodeAgent


class Agent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.llm_cfg = cfg.llm
        self.ideas: list[str] = []
        self.reports: list[Pipeline] = []
        self.failures: list[str] = []
        self.iteration = 0
        self.task_desc = cfg.task_description
        (cfg.agent_workspace_dir / "predictions").mkdir(parents=True, exist_ok=False)
        self.logger = get_logger("Agent", cfg.agent_workspace_dir / "agent.log")
        self.llm_logger = get_logger("LLM (Agent)", cfg.agent_workspace_dir / "llm.log", file_only=True)
        self.hardware = detect_hardware(cfg.execution_device, cfg.execution_max_gpu_count)
        self.logger.info(self.hardware.to_prompt())
        self.is_lower_better = self._query_is_lower_better()
        self.metric_updater = MetricUpdater(cfg, WorstMetricValue(), self)

        evaluator_cfg = deepcopy(cfg)
        self.evaluator = Evaluator(evaluator_cfg, self.is_lower_better)
        # Only the evaluator retains the source data path. Every round sees the same public split.
        cfg.task_input_dir = self.evaluator.public_dir

    def _query_is_lower_better(self) -> bool:
        response = query_llm(self.llm_cfg, messages=[{"role": "system", "content": f"""
You are the metric designer for a local business ML task.
<task_desc>{self.task_desc}</task_desc>
Read the primary metric and optimization direction in the business requirements.
Return <is_lower_better>True</is_lower_better> for minimize, False for maximize,
and <explanation>your rationale</explanation>. Do not invent a different metric.
"""}], required_fields=["is_lower_better"],
            check_fn=lambda result: result["is_lower_better"][0].strip().lower() in {"true", "false"},
            logger=self.llm_logger)
        return response["is_lower_better"][0].strip().lower() == "true"

    def _get_report_from_id(self, id: str) -> Pipeline | None:
        return next((report for report in self.reports if report.id == id), None)

    def _get_ideas_str(self) -> str:
        return "\n".join(f"- {idea}" for idea in self.ideas)

    def _get_reports_str(self, max_reports: int = 30) -> str:
        # Keep both the best results and recent attempts so failures are not forgotten.
        if len(self.reports) <= max_reports:
            reports = self.reports
        else:
            recent = self.reports[-(max_reports // 2):]
            recent_ids = {report.id for report in recent}
            best = sorted((r for r in self.reports if r.id not in recent_ids),
                          key=lambda r: r.metric, reverse=True)[:max_reports - len(recent)]
            reports = best + recent
        failures = "\n".join(self.failures[-20:])
        return "\n".join(report.to_str(full_code=False) for report in reports) + f"\n<failed_attempts>{failures}</failed_attempts>"

    def _context(self) -> str:
        return f"""
<task_desc>{self.task_desc}</task_desc>
<execution_environment>{self.hardware.to_prompt()}
CPU core limit per session: {self.cfg.execution_max_cpu_cores}.
Maximum time per code cell: {self.cfg.execution_timeout} seconds.
</execution_environment>
<ideas>{self._get_ideas_str()}</ideas>
<local_experiments>{self._get_reports_str()}</local_experiments>
The business requirements define the target, primary metric, validation rules,
permitted inputs and output schema. Use supplied local data, optional
existing_solution/existing_code references, and measured experiments from this run.
Do not download datasets, pretrained models or predictions.
When references are provided, adapt their approach and useful code into the first
baseline for the current business requirements. Reference code is supplied as text;
its original files, imports, datasets and artifacts are not automatically available.
Historical scores are unverified: establish a fresh score on the current split.
When both references are absent or empty, start with a simple reproducible baseline.
Keep the evaluator's split fixed,
fit preprocessing on training rows only, and never read hidden validation labels.
Every improvement needs a testable hypothesis and comparable validation evidence.
Treat failed and unscored attempts as lessons, not successful results.
Use CPU when a library lacks support for the selected accelerator.
"""

    def _brainstorm(self):
        prompt = f"""
You are BizMind ML's experiment planner for a local business task.
{self._context()}
Propose at least four feasible solution paths. On the first round prioritize a
cheap baseline and data-quality checks. On later rounds identify weaknesses in
measured local results, avoid repeating failed attempts, and propose controlled
changes with a hypothesis, expected evidence, runtime and memory cost.
Include error analysis, calibration or interpretability when business goals need them.
Respond using repeated tags:
<solution><description>Approach, hypothesis and validation plan</description>
<idea>One concrete implementation step</idea><idea>Another step</idea></solution>
"""
        def validate(response):
            try:
                return len(response["solution"]) >= 4 and all(
                    extract_fields(item, ["description", "idea"]) for item in response["solution"])
            except (ValueError, KeyError):
                return False
        response = query_llm(self.llm_cfg, messages=[{"role": "system", "content": prompt}],
                             required_fields=["solution"], check_fn=validate, logger=self.llm_logger)
        for solution in response["solution"]:
            fields = extract_fields(solution, ["description", "idea"])
            self.ideas.extend(fields["description"] + fields["idea"])

    def _rephrase(self):
        response = query_llm(self.llm_cfg, messages=[{"role": "system", "content": f"""
You are refining BizMind ML's local experiment backlog.
{self._context()}
Merge duplicate ideas and remove disproven or infeasible changes. Preserve useful
technical details and business requirements. Rank actions by measured weaknesses,
expected benefit and cost. Keep at most 20 actionable ideas; preserve a baseline
until a valid local result exists. Respond with repeated <idea>...</idea> tags.
"""}], required_fields=["idea"], logger=self.llm_logger)
        self.ideas = response["idea"][:20]

    def _generate_pipelines(self) -> list[Draft]:
        usable = [r for r in self.reports if not r.metric.is_worst and r.full_code]
        focus = iteration_focus(self.iteration, bool(usable), self.cfg.agent_num_code_agents)
        prompt = f"""
You are designing BizMind ML's next local experiments.
{self._context()}
Round focus: {focus}.
Generate exactly {self.cfg.agent_num_code_agents} independent pipelines.
For baseline, adapt supplied existing_solution/existing_code into a runnable
baseline when available; otherwise use a simple baseline or repair the failed one.
For improve, extend the best measured method with a focused change.
For explore, test a materially different method against the current best.
For mixed, include an extension of the best method and a distinct alternative.
With one worker choose one coherent experiment. Every pipeline must state its
hypothesis, parent result, validation comparison, data loading, target, feature
handling, model parameters, hardware, budget, output format and artifacts.
Use ../input/data/ for local inputs. Supplied existing_code may be adapted as an
unscored reference. Reuse artifacts only from completed, scored local experiments
listed here. Reference text is not a scored parent: use codebase None until a
completed experiment from this run can be selected. Outputs of other pipelines in this round
are unavailable. A model must be trainable using local inputs; no remote resources.
When using neural networks prefer PyTorch and the detected backend.
Respond with repeated blocks:
<pipeline>
<title>Unique descriptive title of at most 30 characters</title>
<description>Detailed reproducible implementation and validation plan</description>
<codebase>Exact ID of a completed scored local experiment, or None</codebase>
<code>Python outline including data loading and prediction output</code>
</pipeline>
"""
        def validate(response):
            try:
                if len(response["pipeline"]) != self.cfg.agent_num_code_agents:
                    return False
                for item in response["pipeline"]:
                    fields = extract_fields(item, ["title", "description", "codebase", "code"])
                    parent = fields["codebase"][0]
                    if parent.lower() != "none" and parent not in {r.id for r in usable}:
                        return False
                return True
            except (ValueError, KeyError, IndexError):
                return False
        response = query_llm(self.llm_cfg, messages=[{"role": "system", "content": prompt}],
                             required_fields=["pipeline"], check_fn=validate, logger=self.llm_logger)
        drafts = []
        for index, item in enumerate(response["pipeline"]):
            fields = extract_fields(item, ["title", "description", "codebase", "code"])
            parent = fields["codebase"][0]
            parent = None if parent.lower() == "none" else parent
            report = self._get_report_from_id(parent) if parent else None
            # Runtime-generated IDs prevent title collisions and keep all paths inside the run.
            drafts.append(Draft(
                id=f"round-{self.iteration + 1:03d}-experiment-{index + 1:02d}",
                title=fields["title"][0], description=fields["description"][0],
                codebase=parent, codebase_content=report.full_code if report else None,
                code=fields["code"][0]))
        return drafts

    def _setup_coder_workspace(self, draft: Draft, base_dir: Path):
        (base_dir / "input/data").mkdir(parents=True, exist_ok=False)
        (base_dir / "working").mkdir(parents=True, exist_ok=False)
        copytree(self.cfg.task_input_dir, base_dir / "input/data", use_symlinks=True)
        if draft.codebase:
            report = self._get_report_from_id(draft.codebase)
            if report is None or report.output_dir is None:
                raise ValueError(f"Local parent experiment not found: {draft.codebase}")
            copytree(report.output_dir, base_dir / "working", use_symlinks=False)

    def _start_coder(self, draft: Draft) -> Pipeline:
        cfg = deepcopy(self.cfg)
        self._setup_coder_workspace(draft, cfg.agent_workspace_dir / draft.id)
        cfg.agent_workspace_dir = cfg.agent_workspace_dir / draft.id / "working"
        coder = CodeAgent(cfg=cfg, draft=draft, is_lower_better=self.is_lower_better,
                          metric_updater=self.metric_updater, evaluator=self.evaluator,
                          env_name=hashlib.md5(draft.id.encode()).hexdigest()[:8])
        return coder.run()

    def _start_coder_with_result(self, draft: Draft, results, index: int):
        try:
            results[index] = self._start_coder(draft)
        except Exception as error:
            message = f"{draft.id} ({draft.title}) failed: {error}"
            self.failures.append(message)
            self.logger.exception(message)

    def _start_coders(self, drafts: list[Draft]) -> list[Pipeline]:
        results = [None] * len(drafts)
        threads = [threading.Thread(target=self._start_coder_with_result, args=(draft, results, index))
                   for index, draft in enumerate(drafts)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return [report for report in results if report is not None]

    def _save_history(self):
        history = [{"id": r.id, "title": r.title, "description": r.description,
                    "score": r.metric.value, "suggestions": r.suggestions,
                    "prediction_file": str(r.prediction) if r.prediction else None}
                   for r in self.reports]
        (self.cfg.agent_workspace_dir / "experiments.json").write_text(
            json.dumps({"experiments": history, "failed_attempts": self.failures}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def run(self):
        with Status("Preparing a fixed local validation split..."):
            self.evaluator.setup()
        for iteration in range(self.cfg.agent_num_iterations):
            self.iteration = iteration
            self.logger.info(f"Local experiment round {iteration + 1}")
            with Status("Learning from local experiments..."):
                self._brainstorm()
                self._rephrase()
                drafts = self._generate_pipelines()
            consumer = threading.Thread(target=self.metric_updater.run)
            consumer.start()
            try:
                self.reports.extend(self._start_coders(drafts))
            finally:
                self.metric_updater.stop()
                consumer.join()
            self._save_history()
        if self.metric_updater.best_metric.is_worst:
            raise RuntimeError(
                f"No valid local experiment after {self.cfg.agent_num_iterations} rounds. "
                f"Inspect {self.cfg.agent_workspace_dir / 'experiments.json'} and worker logs.")
