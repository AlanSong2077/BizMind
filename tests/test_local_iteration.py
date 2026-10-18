"""Exercise local feedback, artifact reuse, and independent evaluation."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import yaml

from helpers import load_classes


class LocalIterationTests(unittest.TestCase):
    def agent_class(self, **dependencies):
        return load_classes(
            "bizmind/agent.py", {"Agent"}, Status=lambda *a: nullcontext(),
            get_logger=Mock(), WorstMetricValue=Mock(), MetricUpdater=Mock(),
            detect_hardware=Mock(return_value=SimpleNamespace(to_prompt=lambda: "CPU")),
            threading=threading, deepcopy=deepcopy, json=json,
            **dependencies,
        )["Agent"]

    def test_two_rounds_feed_local_successes_and_failures_back_to_planning(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = SimpleNamespace(
                llm=None, task_description="Predict local sales", task_id="sales",
                task_input_dir=root, agent_workspace_dir=root / "run",
                agent_prediction_file_name="predictions.csv", agent_num_iterations=2,
                execution_device="cpu", execution_max_gpu_count=0,
            )
            factory = Mock(return_value=SimpleNamespace(public_dir=root / "public", setup=Mock()))
            Agent = self.agent_class(Evaluator=factory)
            report = SimpleNamespace(id="round-1-baseline", title="Baseline", description="Measured baseline",
                                     full_code="print('baseline')", metric=SimpleNamespace(value=0.8),
                                     suggestions="Try regularization", prediction=None)
            seen = []
            with patch.object(Agent, "_query_is_lower_better", return_value=False), \
                 patch.object(Agent, "_brainstorm", lambda a: seen.append((list(a.reports), list(a.failures)))), \
                 patch.object(Agent, "_rephrase"), \
                 patch.object(Agent, "_generate_pipelines", return_value=[]), \
                 patch.object(Agent, "_start_coders", side_effect=[[report], []]):
                agent = Agent(cfg)
                agent.metric_updater.best_metric = SimpleNamespace(is_worst=False)
                agent.failures.append("oversized model ran out of memory")
                agent.run()
            self.assertEqual(seen[0][0], [])
            self.assertEqual(seen[1][0], [report])
            self.assertIn("out of memory", seen[1][1][0])
            factory.return_value.setup.assert_called_once()
            self.assertEqual(cfg.task_input_dir, root / "public")

    def test_exhausted_run_with_no_valid_result_fails_clearly_and_keeps_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = SimpleNamespace(llm=None, task_description="Business task", task_id="sales",
                                  task_input_dir=root, agent_workspace_dir=root / "run",
                                  agent_num_iterations=1, execution_device="cpu", execution_max_gpu_count=0)
            Agent = self.agent_class(Evaluator=Mock(return_value=SimpleNamespace(public_dir=root / "public", setup=Mock())))
            with patch.object(Agent, "_query_is_lower_better", return_value=False), \
                 patch.object(Agent, "_brainstorm"), patch.object(Agent, "_rephrase"), \
                 patch.object(Agent, "_generate_pipelines", return_value=[]), \
                 patch.object(Agent, "_start_coders", return_value=[]):
                agent = Agent(cfg)
                agent.metric_updater.best_metric = SimpleNamespace(is_worst=True)
                with self.assertRaisesRegex(RuntimeError, "No valid local experiment"):
                    agent.run()
            self.assertTrue((root / "run/experiments.json").is_file())

    def test_workspace_reuses_only_completed_local_code_and_artifacts(self):
        # The filesystem helper is loaded with import boundaries removed.
        import ast
        from helpers import ROOT
        path = ROOT / "bizmind/utils/__init__.py"
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.FunctionDef))]
        namespace = {"Path": Path}
        exec(compile(tree, str(path), "exec"), namespace)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, previous = root / "data", root / "previous"
            source.mkdir()
            previous.mkdir()
            (source / "train.csv").write_text("x,y\n1,0\n")
            (previous / "model.bin").write_text("local checkpoint")
            Agent = self.agent_class(copytree=namespace["copytree"])
            agent = Agent.__new__(Agent)
            agent.cfg = SimpleNamespace(task_input_dir=source)
            agent.logger = Mock()
            agent.reports = [SimpleNamespace(id="baseline", output_dir=previous, full_code="train()")]
            draft = SimpleNamespace(id="improvement", codebase="baseline")
            agent._setup_coder_workspace(draft, root / "next")
            self.assertEqual((root / "next/input/data/train.csv").read_text(), "x,y\n1,0\n")
            self.assertEqual((root / "next/working/model.bin").read_text(), "local checkpoint")
            self.assertFalse((root / "next/working/model.bin").is_symlink())

    def test_evaluation_returns_only_a_local_score(self):
        with TemporaryDirectory() as directory:
            cfg = SimpleNamespace(llm=None, task_id="sales", agent_workspace_dir=Path(directory),
                                  agent_prediction_file_name="predictions.csv")
            Evaluator = load_classes(
                "bizmind/evaluate.py", {"Evaluator", "EvaluationResult"},
                dataclass=dataclass, get_logger=Mock(), Conversation=Mock(),
                execute_file=Mock(side_effect=lambda *args: (
                    (Path(directory) / "evaluation/private/eval_report.json").write_text(
                        '{"score": 0.85, "success": true, "message": "ok"}'),
                    SimpleNamespace(success=True))[1]), json=json, math=math,
                threading=threading,
            )["Evaluator"]
            evaluator = Evaluator(cfg, False)
            (evaluator.private_dir / "eval_report.json").write_text(
                '{"score": 999, "success": true, "message": "stale"}')
            result = evaluator.evaluate(Path(directory) / "prediction.csv")
            self.assertEqual(result.score, 0.85)
            self.assertTrue(result.success)
            self.assertFalse(hasattr(result, "rank"))

    def test_local_config_roundtrip_and_rejects_obsolete_or_unknown_options(self):
        class Mixin:
            @classmethod
            def from_dict(cls, values):
                return cls(**values)
        Config = load_classes(
            "bizmind/config/config.py", {"Config", "LLMConfig"},
            dataclass=dataclass, field=field, Path=Path, yaml=yaml,
            DataClassJsonMixin=Mixin, time=time,
        )["Config"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            cfg = Config(task_id="sales", task_input_file=Path("business.md"),
                         agent_prediction_file_name="forecast.csv")
            cfg.to_yaml(path)
            loaded = Config.from_yaml(path)
            self.assertEqual(loaded.task_id, "sales")
            self.assertEqual(loaded.task_input_file, Path("business.md"))
            self.assertEqual(loaded.agent_prediction_file_name, "forecast.csv")
            path.write_text("agent: {unknown_option: true}\n")
            with self.assertRaises(ValueError):
                Config.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
