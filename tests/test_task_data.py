"""Real local files exercise custom task loading and input staging."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import runpy
import sys
import unittest
from unittest.mock import Mock, patch

from helpers import ROOT
from bizmind.config import Config

from bizmind.task_input import load_custom_task, stage_custom_data


class TaskDataTests(unittest.TestCase):
    def config(self, root, **values):
        return Config(task_input_file=root / "task.md", task_train_data=None,
                               task_predict_data=None, agent_workspace_dir=root / "run",
                               task_input_dir=None, task_id="local", **values)

    def test_yaml_combines_business_and_file_relative_data_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.csv").write_text("x,y\n1,0\n")
            (root / "test.csv").write_text("x\n2\n")
            (root / "task.yaml").write_text(
                "background: 客户运营\nobjective: 预测流失\nevaluation:\n  metric: roc_auc\n"
                "data:\n  train: ./train.csv\n  predict: ./test.csv\n", encoding="utf-8")
            cfg = self.config(root)
            cfg.task_input_file = root / "task.yaml"
            text = load_custom_task(cfg)
            self.assertIn("客户运营", text)
            self.assertEqual(cfg.task_train_data, (root / "train.csv").resolve())
            staged = stage_custom_data(cfg, text)
            self.assertEqual((cfg.task_input_dir / "train_data/train.csv").read_text(), "x,y\n1,0\n")
            self.assertEqual((cfg.task_input_dir / "predict_data/test.csv").read_text(), "x\n2\n")
            self.assertIn("train_data/train.csv", staged)
            self.assertNotIn(str(root), staged)

    def test_explicit_sources_override_yaml_and_identical_names_do_not_collide(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for role in ("a", "b"):
                (root / role).mkdir()
                (root / role / "data.csv").write_text(role)
            (root / "task.yaml").write_text("objective: regression\ndata: {train: missing, predict: missing}\n")
            cfg = self.config(root)
            cfg.task_input_file = root / "task.yaml"
            cfg.task_train_data, cfg.task_predict_data = root / "a/data.csv", root / "b/data.csv"
            text = stage_custom_data(cfg, load_custom_task(cfg))
            self.assertEqual((cfg.task_input_dir / "train_data/data.csv").read_text(), "a")
            self.assertEqual((cfg.task_input_dir / "predict_data/data.csv").read_text(), "b")
            self.assertIn("predict_data/data.csv", text)

    def test_directories_are_supported_without_copying_or_modifying_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for role in ("train", "predict"):
                (root / role).mkdir()
                (root / role / "part.csv").write_text(role)
            cfg = self.config(root)
            cfg.task_train_data, cfg.task_predict_data = root / "train", root / "predict"
            stage_custom_data(cfg, "Business objective")
            self.assertTrue((cfg.task_input_dir / "train_data").is_symlink())
            self.assertEqual((root / "train/part.csv").read_text(), "train")

    def test_missing_pair_and_nested_output_are_rejected_before_staging(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            cfg.task_train_data = root
            with self.assertRaisesRegex(ValueError, "both"):
                stage_custom_data(cfg, "task")
            cfg.task_predict_data = root
            with self.assertRaisesRegex(ValueError, "inside"):
                stage_custom_data(cfg, "task")
            self.assertFalse(cfg.agent_workspace_dir.exists())

    def test_legacy_directory_is_preserved_when_no_explicit_sources_are_given(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config(root)
            cfg.task_input_dir = root / "data"
            cfg.task_input_dir.mkdir()
            self.assertEqual(stage_custom_data(cfg, "Original description"), "Original description")
            self.assertEqual(cfg.task_input_dir, (root / "data").resolve())

    def test_overlapping_roles_are_rejected_to_keep_training_labels_out_of_prediction_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/train.csv").write_text("x,y\n1,0\n")
            cfg = self.config(root)
            cfg.task_train_data, cfg.task_predict_data = root / "data/train.csv", root / "data"
            with self.assertRaisesRegex(ValueError, "overlap"):
                stage_custom_data(cfg, "Task")
            self.assertFalse(cfg.agent_workspace_dir.exists())

    def test_cli_supports_task_yaml_without_data_dir_and_manual_device_override(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.csv").write_text("x,y\n1,0\n")
            (root / "predict.csv").write_text("x\n2\n")
            task = root / "business.yaml"
            task.write_text("objective: prediction\ndata: {train: missing.csv, predict: missing.csv}\n")
            cfg = self.config(root)
            cfg.execution_device = "auto"
            agent = Mock()
            with patch.dict(sys.modules, {
                "bizmind.config": SimpleNamespace(Config=SimpleNamespace(from_yaml=Mock(return_value=cfg))),
                "bizmind.agent": SimpleNamespace(Agent=agent),
            }), patch.object(sys, "argv", ["main.py", "--task-file", str(task),
                                           "--train-data", str(root / "train.csv"),
                                           "--predict-data", str(root / "predict.csv"), "--device", "cpu"]):
                runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
            self.assertEqual(cfg.task_id, "local")
            self.assertEqual(cfg.execution_device, "cpu")
            self.assertTrue((cfg.task_input_dir / "train_data/train.csv").exists())
            agent.return_value.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
