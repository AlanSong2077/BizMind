"""Verify business and device requirements reach planning and execution prompts."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from helpers import load_classes


class TrainingContextTests(unittest.TestCase):
    def test_coder_receives_business_data_device_limits_and_local_feedback_rules(self):
        cls = load_classes("bizmind/coder.py", {"CodeAgent"}, generate=Mock(return_value="Data overview"))["CodeAgent"]
        coder = cls.__new__(cls)
        coder.cfg = SimpleNamespace(agent_workspace_dir=Path("/tmp/example/working"), execution_max_cpu_cores=2,
                                    task_description="Business target with local data", execution_timeout=120)
        coder.draft = SimpleNamespace(description="Train a model", code="print('train')", codebase_content=None)
        coder.prediction_name = "predictions.csv"
        coder.llm = Mock()
        coder.jupyter_session = SimpleNamespace(hardware=SimpleNamespace(
            to_prompt=lambda: "Selected training device: mps; Apple Metal shared memory."))
        coder._post_initial_message()
        prompt = coder.llm.add_message.call_args.kwargs["content"]
        for value in ("Selected training device: mps", "CPU core limit: 2", "120 seconds",
                      "Business target with local data", "../input/data/", "independent local evaluator",
                      "Do not download", "BIZMIND_DEVICE"):
            self.assertIn(value, prompt)
        self.assertNotIn("A6000", prompt)

    def test_evaluator_keeps_source_path_and_coders_receive_only_public_split(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = SimpleNamespace(llm=None, task_description="Business task", task_id="local",
                                  task_input_dir=root / "original", agent_workspace_dir=root / "run",
                                  agent_prediction_file_name="predictions.csv", execution_device="cpu",
                                  execution_max_gpu_count=0)
            public = root / "run/evaluation/public"
            factory = Mock(return_value=SimpleNamespace(public_dir=public))
            Agent = load_classes("bizmind/agent.py", {"Agent"}, get_logger=Mock(), MetricUpdater=Mock(),
                                 WorstMetricValue=Mock(), deepcopy=deepcopy, Evaluator=factory,
                                 detect_hardware=Mock(return_value=SimpleNamespace(to_prompt=lambda: "CPU")))["Agent"]
            with patch.object(Agent, "_query_is_lower_better", return_value=True):
                Agent(cfg)
            self.assertEqual(factory.call_args.args[0].task_input_dir, root / "original")
            self.assertEqual(cfg.task_input_dir, public)


if __name__ == "__main__":
    unittest.main()
