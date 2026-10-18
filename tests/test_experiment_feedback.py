"""Regression tests for fixed local scoring and multi-round result handling."""
import ast
from copy import deepcopy
from dataclasses import dataclass, field
from functools import total_ordering
import json
import math
from pathlib import Path
import queue
import runpy
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from helpers import ROOT, load_classes


metrics = load_classes("bizmind/utils/metric.py", {"MetricValue", "WorstMetricValue"},
                       dataclass=dataclass, field=field, total_ordering=total_ordering,
                       np=np, DataClassJsonMixin=object)
MetricValue, WorstMetricValue = metrics["MetricValue"], metrics["WorstMetricValue"]


def real_script_executor():
    tree = ast.parse((ROOT / "bizmind/utils/execute.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(dataclass=dataclass, Path=Path, subprocess=subprocess, sys=sys,
                     Optional=__import__("typing").Optional, process_backspace_chars=lambda value: value)
    exec(compile(tree, "execute.py", "exec"), namespace)
    return namespace["execute_file"]


class ExperimentFeedbackTests(unittest.TestCase):
    def test_strategy_establishes_baseline_then_balances_improvement_and_exploration(self):
        focus = runpy.run_path(str(ROOT / "bizmind/strategy.py"))["iteration_focus"]
        self.assertEqual(focus(0, False, 1), "baseline")
        self.assertEqual(focus(3, False, 4), "baseline")
        self.assertEqual(focus(1, True, 1), "improve")
        self.assertEqual(focus(2, True, 1), "explore")
        self.assertEqual(focus(1, True, 2), "mixed")

    def test_metric_queue_drains_all_results_and_snapshots_files_across_rounds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "predictions").mkdir()
            cfg = SimpleNamespace(agent_workspace_dir=root, start_time=time.time(),
                                  agent_prediction_file_name="predictions.csv")
            Updater = load_classes("bizmind/agent.py", {"MetricUpdater"}, queue=queue, time=time,
                                   json=json, WorstMetricValue=WorstMetricValue, get_logger=Mock())["MetricUpdater"]
            updater = Updater(cfg, WorstMetricValue())
            prediction = root / "worker.csv"
            for score in (0.8, 0.9):
                prediction.write_text(str(score))
                updater.post(MetricValue(score, maximize=True), prediction)
                prediction.write_text("overwritten by later trial")
                updater.stop()
                consumer = threading.Thread(target=updater.run)
                consumer.start()
                consumer.join(timeout=2)
                self.assertFalse(consumer.is_alive())
            self.assertEqual(updater.best_metric.value, 0.9)
            self.assertEqual((root / "predictions/predictions.csv").read_text(), "0.9")
            self.assertEqual(json.loads((root / "best.json").read_text())["score"], 0.9)
            self.assertEqual(len(list((root / "predictions").glob('*_predictions.csv'))), 2)

    def evaluator(self, root):
        cfg = SimpleNamespace(agent_workspace_dir=root, agent_prediction_file_name="predictions.csv", llm=None)
        Evaluator = load_classes("bizmind/evaluate.py", {"Evaluator", "EvaluationResult"},
                                 dataclass=dataclass, json=json, math=math, threading=threading,
                                 get_logger=Mock(), Conversation=Mock(), execute_file=real_script_executor(),
                                 generate=Mock(return_value="Local data"), shutil=shutil,
                                 execute=load_classes("bizmind/utils/execute.py", {"execute"},
                                                      execute_file=real_script_executor())["execute"])["Evaluator"]
        return Evaluator(cfg, False)

    def test_two_script_budget_finishes_immediately_after_successful_real_evaluation(self):
        with TemporaryDirectory() as directory:
            evaluator = self.evaluator(Path(directory))
            evaluator.cfg.agent_num_iterations_code_agents = 2
            evaluator.cfg.task_description = "Local regression with MSE"
            evaluator.cfg.task_input_dir = Path(directory)
            evaluator.llm.query.side_effect = [
                {"current_file": ["split_dataset.py"], "code": [
                    "from pathlib import Path\nPath('public/validation_sample_predictions.csv').write_text('id,pred\\n1,0\\n')"]},
                {"current_file": ["evaluate.py"], "code": [
                    "from pathlib import Path\nPath('private/eval_report.json').write_text('{\"success\":true,\"score\":0.5}')"]},
            ]
            evaluator.setup()
            self.assertEqual(evaluator.llm.query.call_count, 2)
            self.assertTrue(evaluator.eval_script_path.is_file())

    def test_real_scoring_script_is_serialized_across_concurrent_experiments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = self.evaluator(root)
            evaluator.eval_script_path.write_text('''import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
for key in ('public_dir', 'private_dir', 'pred'):
    p.add_argument('--' + key)
a = p.parse_args()
score = float(Path(a.pred).read_text())
time.sleep(0.04)
(Path(a.private_dir) / 'eval_report.json').write_text(json.dumps({'score':score,'success':True,'message':''}))
''')
            results = {}
            def worker(index):
                prediction = root / f"prediction-{index}.txt"
                prediction.write_text(str(index / 10))
                results[index] = evaluator.evaluate(prediction)
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual({i: result.score for i, result in results.items()}, {i: i / 10 for i in range(4)})
            self.assertTrue(all(result.success for result in results.values()))

    def test_failed_split_with_partial_files_cannot_pass_setup(self):
        with TemporaryDirectory() as directory:
            evaluator = self.evaluator(Path(directory))
            evaluator.cfg.agent_num_iterations_code_agents = 2
            evaluator.cfg.task_description = "Local regression"
            evaluator.cfg.task_input_dir = Path(directory)
            evaluator.llm.query.side_effect = [
                {"current_file": ["split_dataset.py"], "code": [
                    "from pathlib import Path\nPath('public/validation_sample_predictions.csv').write_text('partial')\nraise RuntimeError('split incomplete')"]},
                {"current_file": ["evaluate.py"], "code": [
                    "from pathlib import Path\nPath('private/eval_report.json').write_text('{\"success\":true,\"score\":0.5}')"]},
            ]
            with self.assertRaisesRegex(RuntimeError, "setup exceeded"):
                evaluator.setup()

    def test_stale_report_and_nonfinite_scores_cannot_become_successful_results(self):
        with TemporaryDirectory() as directory:
            evaluator = self.evaluator(Path(directory))
            report = evaluator.private_dir / "eval_report.json"
            report.write_text('{"success":true,"score":999}')
            evaluator.eval_script_path.write_text("print('no report generated')")
            result = evaluator.evaluate(Path(directory) / "pred.csv")
            self.assertFalse(result.success)
            self.assertIsNone(result.score)
            for score in ("NaN", "Infinity", "true", "null", '"0.9"'):
                report.write_text('{"success":true,"score":' + score + '}')
                self.assertFalse(evaluator._read_report().success)

    def test_single_iteration_budget_executes_a_cell_and_shutdown_is_guaranteed(self):
        cls = load_classes("bizmind/coder.py", {"CodeAgent"}, time=time, shutil=shutil)["CodeAgent"]
        with TemporaryDirectory() as directory:
            coder = cls.__new__(cls)
            coder.cfg = SimpleNamespace(agent_num_iterations_code_agents=1, agent_time_limit_code_agents=60,
                                        agent_workspace_dir=Path(directory))
            coder.start_time = time.time()
            coder.best_prediction_path = Path(directory) / "missing.csv"
            coder._post_initial_message = Mock()
            coder._generate_report = Mock(return_value="local report")
            coder.llm = Mock()
            coder.logger = Mock()
            coder.llm.query.return_value = {"code": ["print('baseline')"], "goal": ["baseline"],
                                           "prediction_file": ["None"], "validation_prediction_file": ["None"]}
            coder.jupyter_session = Mock()
            coder.jupyter_session.append_cell.return_value = SimpleNamespace(
                success=True, output="baseline", error=None, execution_time=0.01)
            self.assertEqual(coder.run(), "local report")
            coder.jupyter_session.append_cell.assert_called_once()
            coder.jupyter_session.shutdown.assert_called_once()
            coder.jupyter_session.reset_mock()
            coder.llm.query.side_effect = RuntimeError("model service unavailable")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                coder.run()
            coder.jupyter_session.shutdown.assert_called_once()

    def test_later_regression_does_not_overwrite_best_prediction(self):
        cls = load_classes("bizmind/coder.py", {"CodeAgent"}, Path=Path, shutil=shutil, deepcopy=deepcopy,
                           MetricValue=MetricValue)["CodeAgent"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            coder = cls.__new__(cls)
            coder.cfg = SimpleNamespace(agent_workspace_dir=root)
            coder.is_lower_better = False
            coder.best_metric = WorstMetricValue()
            coder.best_prediction_path = root / "best.csv"
            coder.jupyter_session = SimpleNamespace(cells="best executed code")
            coder.metric_updater = Mock()
            coder.evaluator = Mock()
            coder.evaluator.evaluate.side_effect = [SimpleNamespace(success=True, score=0.9),
                                                    SimpleNamespace(success=True, score=0.7)]
            prediction, validation = root / "pred.csv", root / "valid.csv"
            response = {"prediction_file": [prediction.name], "validation_prediction_file": [validation.name]}
            validation.write_text("validation predictions")
            prediction.write_text("best predictions")
            coder._evaluate_predictions(response)
            prediction.write_text("regressed predictions")
            coder._evaluate_predictions(response)
            self.assertEqual(coder.best_prediction_path.read_text(), "best predictions")
            self.assertEqual(coder.best_metric.value, 0.9)
            coder.metric_updater.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
