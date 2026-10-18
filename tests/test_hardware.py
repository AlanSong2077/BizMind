"""Hardware detection and kernel launch tests without GPUs or notebook processes."""
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def fake_torch(cuda_count=0, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_count > 0,
            device_count=lambda: cuda_count,
            get_device_properties=lambda index: SimpleNamespace(
                name=f"GPU logical {index}", total_memory=(index + 1) * 8 * 1024 ** 3),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def load_session_class():
    tree = ast.parse((ROOT / "bizmind/utils/jupyter_session.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "JupyterSession"]
    namespace = dict(
        Config=object, List=list, Tuple=tuple, KernelManager=Mock(),
        BlockingKernelClient=object, KernelSpecManager=Mock(), ExecutionResult=object,
        JupyterNotebook=Mock(), atexit=Mock(), os=os, json=json, Path=Path,
        tempfile=tempfile, sys=sys, platform=platform, shutil=shutil,
        psutil=SimpleNamespace(cpu_percent=lambda **kwargs: [20, 5, 10]),
        subprocess=Mock(), time=lambda: 0,
    )
    exec(compile(tree, "jupyter_session.py", "exec"), namespace)
    return namespace


class HardwareDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detect = runpy.run_path(str(ROOT / "bizmind/utils/hardware.py"))["detect_hardware"]

    def test_cpu_without_torch_does_not_need_nvidia_tools(self):
        with patch.dict(sys.modules, {"torch": None}), patch.dict(os.environ, {}, clear=True):
            hardware = self.detect()
        self.assertEqual(hardware.device, "cpu")
        self.assertEqual(hardware.cuda_visible_devices, ())
        self.assertIn("cpu", hardware.to_prompt().lower())

    def test_cuda_respects_cap_and_reports_actual_memory(self):
        with patch.dict(sys.modules, {"torch": fake_torch(cuda_count=3)}), \
             patch.dict(os.environ, {}, clear=True):
            hardware = self.detect(max_gpu_count=2)
        self.assertEqual(hardware.device, "cuda")
        self.assertEqual(hardware.cuda_visible_devices, ("0", "1"))
        self.assertEqual(hardware.memory_gb, (8.0, 16.0))
        self.assertIn("GPU logical 1", hardware.to_prompt())

    def test_cuda_preserves_physical_and_uuid_visibility_tokens(self):
        for visible, expected in [("3,1", ("3",)), ("GPU-abcd,GPU-efgh", ("GPU-abcd",)),
                                  ("MIG-GPU-abcd/1/0", ("MIG-GPU-abcd/1/0",))]:
            with self.subTest(visible=visible), \
                 patch.dict(sys.modules, {"torch": fake_torch(cuda_count=2)}), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}, clear=True):
                hardware = self.detect()
            self.assertEqual(hardware.cuda_visible_devices, expected)

    def test_explicitly_hidden_cuda_is_never_reenabled(self):
        for visible in ["", "-1", "invalid"]:
            with self.subTest(visible=visible), \
                 patch.dict(sys.modules, {"torch": fake_torch(cuda_count=2)}), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}, clear=True):
                hardware = self.detect()
            self.assertEqual(hardware.device, "cpu")
            self.assertEqual(hardware.cuda_visible_devices, ())

    def test_mac_uses_available_mps_without_cuda(self):
        with patch.dict(sys.modules, {"torch": fake_torch(mps=True)}), \
             patch.object(platform, "system", return_value="Darwin"):
            hardware = self.detect()
        self.assertEqual(hardware.device, "mps")
        self.assertIn("MPS", hardware.to_prompt())
        self.assertIn("CUDA", hardware.to_prompt())

    def test_zero_gpu_limit_and_manual_cpu_disable_acceleration(self):
        with patch.dict(sys.modules, {"torch": fake_torch(cuda_count=2, mps=True)}):
            self.assertEqual(self.detect(max_gpu_count=0).device, "cpu")
            self.assertEqual(self.detect(device="cpu").device, "cpu")

    def test_mps_reports_recommended_shared_memory_when_available(self):
        torch = fake_torch(mps=True)
        torch.mps = SimpleNamespace(recommended_max_memory=lambda: 12 * 1024 ** 3)
        with patch.dict(sys.modules, {"torch": torch}), \
             patch.object(platform, "system", return_value="Darwin"):
            hardware = self.detect()
        self.assertEqual(hardware.memory_gb, (12.0,))
        self.assertIn("12.0 GiB", hardware.to_prompt())
        self.assertIn("not reserved", hardware.to_prompt())

    def test_unavailable_explicit_device_and_invalid_settings_fail_clearly(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "cuda"):
                self.detect(device="cuda")
        with self.assertRaises(ValueError):
            self.detect(device="tpu")
        with self.assertRaises(ValueError):
            self.detect(max_gpu_count=-1)


class KernelLaunchTests(unittest.TestCase):
    def test_mac_kernel_uses_current_python_without_taskset_or_global_install(self):
        namespace = load_session_class()
        namespace["detect_hardware"] = lambda *args, **kwargs: SimpleNamespace(
            device="mps", cuda_visible_devices=(), to_prompt=lambda: "MPS")
        cfg = SimpleNamespace(agent_workspace_dir=Path("/tmp"), execution_device="auto",
                              execution_max_gpu_count=1, execution_max_cpu_cores=2)
        with patch.object(platform, "system", return_value="Darwin"), \
             patch.object(shutil, "which", side_effect=AssertionError("Mac must not query taskset")):
            session = namespace["JupyterSession"](cfg, "test-mps")
        try:
            kernel_dir = Path(session._kernel_dir.name)
            spec = json.loads((kernel_dir / "test-mps/kernel.json").read_text())
            self.assertEqual(spec["argv"][0], sys.executable)
            self.assertEqual(spec["env"]["BIZMIND_DEVICE"], "mps")
            self.assertEqual(spec["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(spec["env"]["PYTORCH_ENABLE_MPS_FALLBACK"], "1")
            namespace["KernelSpecManager"].return_value.install_kernel_spec.assert_not_called()
        finally:
            session.shutdown()

    def test_linux_taskset_only_uses_permitted_cores(self):
        namespace = load_session_class()
        namespace["detect_hardware"] = lambda *args, **kwargs: SimpleNamespace(
            device="cpu", cuda_visible_devices=(), to_prompt=lambda: "CPU")
        cfg = SimpleNamespace(agent_workspace_dir=Path("/tmp"), execution_device="auto",
                              execution_max_gpu_count=1, execution_max_cpu_cores=2)
        with patch.object(platform, "system", return_value="Linux"), \
             patch.object(shutil, "which", return_value="/usr/bin/taskset"), \
             patch.object(os, "sched_getaffinity", return_value={0, 2}, create=True):
            session = namespace["JupyterSession"](cfg, "test-affinity")
        try:
            spec = json.loads((Path(session._kernel_dir.name) / "test-affinity/kernel.json").read_text())
            self.assertEqual(spec["argv"][:4], ["/usr/bin/taskset", "-c", "2,0", sys.executable])
            self.assertEqual(spec["env"]["CUDA_VISIBLE_DEVICES"], "")
        finally:
            session.shutdown()

    def test_failed_kernel_start_cleans_temporary_spec(self):
        namespace = load_session_class()
        namespace["detect_hardware"] = lambda *args, **kwargs: SimpleNamespace(
            device="cpu", cuda_visible_devices=(), to_prompt=lambda: "CPU")
        namespace["KernelManager"].return_value.start_kernel.side_effect = RuntimeError("cannot start")
        session = namespace["JupyterSession"].__new__(namespace["JupyterSession"])
        cfg = SimpleNamespace(agent_workspace_dir=Path("/tmp"), execution_device="auto",
                              execution_max_gpu_count=0, execution_max_cpu_cores=2)
        with self.assertRaisesRegex(RuntimeError, "cannot start"):
            session.__init__(cfg, "failed-kernel")
        self.assertFalse(Path(session._kernel_dir.name).exists())

    def test_linux_without_taskset_still_runs_and_preserves_gpu_mapping(self):
        namespace = load_session_class()
        namespace["detect_hardware"] = lambda *args, **kwargs: SimpleNamespace(
            device="cuda", cuda_visible_devices=("GPU-one",), to_prompt=lambda: "CUDA")
        cfg = SimpleNamespace(agent_workspace_dir=Path("/tmp"), execution_device="auto",
                              execution_max_gpu_count=1, execution_max_cpu_cores=2)
        with patch.object(platform, "system", return_value="Linux"), \
             patch.object(shutil, "which", return_value=None):
            session = namespace["JupyterSession"](cfg, "test-cuda")
        try:
            spec = json.loads((Path(session._kernel_dir.name) / "test-cuda/kernel.json").read_text())
            self.assertEqual(spec["argv"][0], sys.executable)
            self.assertEqual(spec["env"]["CUDA_VISIBLE_DEVICES"], "GPU-one")
        finally:
            session.shutdown()


class ScriptExecutionTests(unittest.TestCase):
    def test_script_uses_current_interpreter_and_preserves_arguments(self):
        tree = ast.parse((ROOT / "bizmind/utils/execute.py").read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module.startswith("bizmind"))]
        namespace = {"process_backspace_chars": lambda text: text}
        exec(compile(tree, "execute.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            script = workspace / "show_runtime.py"
            script.write_text("import json,sys; print(json.dumps([sys.executable,sys.argv[1]]))")
            result = namespace["execute_file"](workspace, script, ["argument with spaces"])
        self.assertTrue(result.success, result.error)
        self.assertEqual(json.loads(result.output), [sys.executable, "argument with spaces"])


if __name__ == "__main__":
    unittest.main()
