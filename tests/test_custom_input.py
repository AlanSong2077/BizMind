"""Business task input and local CLI checks without model calls."""
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bizmind.config import Config
from helpers import ROOT


class CustomInputTests(unittest.TestCase):
    def run_main(self, directory, input_file=None, args=()):
        root = Path(directory)
        source = root / "data"
        source.mkdir(exist_ok=True)
        cfg = Config(task_id="local-task", task_input_dir=source,
                     task_input_file=input_file, agent_workspace_dir=root / "runs")
        agent = Mock()
        with patch("bizmind.config.Config.from_yaml", return_value=cfg), \
             patch.dict(sys.modules, {"bizmind.agent": SimpleNamespace(Agent=agent)}), \
             patch.object(sys, "argv", ["main.py", *args]):
            runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
        return cfg, agent

    def test_business_text_reaches_the_agent_unchanged(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "business.md"
            text = "# 业务背景\n客户流失预测\n# 指标\nPR-AUC，越高越好\n# 约束\n按时间划分。"
            path.write_text(text, encoding="utf-8")
            cfg, agent = self.run_main(directory, input_file=path)
            self.assertEqual(cfg.task_description, text)
            agent.return_value.run.assert_called_once()

    def test_cli_overrides_task_file_and_local_task_id(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "custom.md"
            path.write_text("Business goal and RMSE metric")
            cfg, _ = self.run_main(directory, input_file=Path("missing.md"),
                                   args=("--task-file", str(path), "--task-id", "sales"))
            self.assertEqual(cfg.task_description, "Business goal and RMSE metric")
            self.assertEqual(cfg.task_id, "sales")
            self.assertEqual(cfg.agent_workspace_dir.parent.name, "sales")

    def test_missing_or_empty_task_file_fails_before_agent_start(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "task.input_file"):
                self.run_main(directory)
            with self.assertRaises(FileNotFoundError):
                self.run_main(directory, input_file=Path(directory) / "missing.md")
            empty = Path(directory) / "empty.md"
            empty.write_text(" \n")
            with self.assertRaisesRegex(ValueError, "empty"):
                self.run_main(directory, input_file=empty)

    def test_config_roundtrip_keeps_local_inputs_budget_and_llm_parameters(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            cfg = Config(task_input_file=Path("business.md"), task_train_data=Path("train.csv"),
                         task_predict_data=Path("predict.csv"), execution_device="cpu")
            cfg.llm.params = {"timeout": 90}
            cfg.to_yaml(path)
            actual = Config.from_yaml(path)
            self.assertEqual(actual.task_input_file, Path("business.md"))
            self.assertEqual(actual.task_train_data, Path("train.csv"))
            self.assertEqual(actual.task_predict_data, Path("predict.csv"))
            self.assertEqual(actual.execution_device, "cpu")
            self.assertEqual(actual.llm.params, {"timeout": 90})

    def test_invalid_identifiers_and_budgets_are_rejected(self):
        for values in ({"task_id": "../escape"}, {"agent_prediction_file_name": "../pred.csv"},
                       {"agent_num_code_agents": 0}, {"agent_num_iterations_code_agents": -1},
                       {"execution_device": "unknown"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values)


if __name__ == "__main__":
    unittest.main()
