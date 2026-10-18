import tempfile
import atexit
import json
import os
import psutil
import platform
import shutil
import sys
from time import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from jupyter_client import KernelManager, BlockingKernelClient
from jupyter_client.kernelspec import KernelSpecManager
from bizmind.experiments import Draft
from bizmind.config import Config, LLMConfig
from bizmind.utils import query_llm, process_backspace_chars
from bizmind.utils.hardware import detect_hardware


@dataclass
class ExecutionResult:
    success: bool
    llm_terminated: bool
    timeout: bool
    output: str
    error: str
    execution_time: float

@dataclass
class CodeCell:
    code: str
    success: bool

class JupyterNotebook:
    cells: list[CodeCell]

    def __init__(self, cfg: Config, cells: list[CodeCell]):
        self.cfg = cfg
        self.cells = cells

    def append(self, code: str, success: bool):
        self.cells.append(CodeCell(code=code, success=success))

    def get_notebook_code(self) -> str:
        cells_str = ""
        for cell in self.cells:
            if cell.success:
                cells_str += f"<cell>\n{cell.code}\n</cell>\n\n"

        prompt = f"""
You are BizMind ML's experiment archivist, converting an executed local notebook
into a reproducible Python script for the best measured experiment.
Preserve local input paths, validation logic, random seeds and business output
schema. Do not invent results, add remote resources or change the experiment.
You should only retain code from cells that were actually executed and required by later cells that produce final results. Remove any unnecessary code and eliminate deprecate code that was replaced by later versions and duplicate implementations where only the final version is used. You extraction should be on cell-level. You should only determine each cell should be retained and rearrange the order of them. Do not change the content of each cell. You should provide a self-contained full python code that can be run as a standalone .py file. Your final code should be contained in a single code tag.

{cells_str}

Now, please provide the assembled python file in the following format:

<summary>
Your summary of the code. Describe the pipeline and model structures.
</summary>

<code>
The assembled python code. Do not include any additional text or explanations. Do not wrap the code in a markdown code block.
</code>
"""
        def check_fn(response: dict):
            return len(response["code"]) == 1

        result = query_llm(self.cfg.llm, [{"role": "system", "content": prompt}], required_fields=["code"], check_fn=check_fn)
        return result["code"][0]


class JupyterSession:
    def __init__(self, cfg: Config, env_name: str):
        self.cfg = cfg
        self.cells = JupyterNotebook(cfg=cfg, cells=[])
        self.env_name = env_name

        self.selected_gpu_ids, self.selected_cpu_cores = self._select_optimal_resources()

        kernel_name = self._setup_conda_environment()

        self.km: KernelManager = KernelManager(
            kernel_name=kernel_name, kernel_spec_manager=self._kernel_spec_manager)
        try:
            self.km.start_kernel(cwd=self.cfg.agent_workspace_dir)
            self.kc: BlockingKernelClient = self.km.client()
            self.kc.start_channels()
            self.kc.wait_for_ready()
        except Exception:
            self._safe_shutdown()
            raise

        atexit.register(self._safe_shutdown)

    def shutdown(self):
        self._safe_shutdown()

    def _get_cpu_core_utilization(self) -> List[Tuple[int, float]]:
        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        available_cores = set(range(len(cpu_percent)))
        if hasattr(os, "sched_getaffinity"):
            try:
                available_cores &= os.sched_getaffinity(0)
            except OSError:
                pass
        return [(i, usage) for i, usage in enumerate(cpu_percent) if i in available_cores]

    def _select_optimal_resources(self) -> Tuple[List[str], List[int]]:
        """Select usable accelerators and lightly loaded permitted CPU cores."""
        self.hardware = detect_hardware(
            getattr(self.cfg, "execution_device", "auto"), self.cfg.execution_max_gpu_count)
        selected_gpu_ids = list(self.hardware.cuda_visible_devices)
        cpu_usage = self._get_cpu_core_utilization()
        cpu_usage.sort(key=lambda x: x[1])
        selected_cpu_cores = [core_id for core_id, _ in cpu_usage[:self.cfg.execution_max_cpu_cores]]

        print(self.hardware.to_prompt())
        print(f"Selected CPU cores: {selected_cpu_cores}")

        return selected_gpu_ids, selected_cpu_cores

    def _setup_conda_environment(self) -> str:
        """Keep this session's kernelspec local; use the running environment."""
        kname = self.env_name
        kernel_env = {
            "CUDA_VISIBLE_DEVICES": ",".join(self.selected_gpu_ids),
            "BIZMIND_DEVICE": self.hardware.device,
        }
        if self.hardware.device == "mps":
            kernel_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if self.cfg.execution_max_cpu_cores > 0:
            limit = str(self.cfg.execution_max_cpu_cores)
            kernel_env.update({key: limit for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
        kernel_argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        taskset = shutil.which("taskset") if platform.system() == "Linux" else None
        if self.selected_cpu_cores and taskset:
            cpu_mask = ",".join(map(str, self.selected_cpu_cores))
            kernel_argv = [taskset, "-c", cpu_mask] + kernel_argv
        self._kernel_dir = tempfile.TemporaryDirectory(prefix="bizmind-kernel-")
        spec_dir = Path(self._kernel_dir.name) / kname
        spec_dir.mkdir()
        kernel_spec = {"argv": kernel_argv, "display_name": kname,
                       "language": "python", "env": kernel_env}
        (spec_dir / "kernel.json").write_text(json.dumps(kernel_spec, indent=2), encoding="utf-8")
        self._kernel_spec_manager = KernelSpecManager(
            kernel_dirs=[self._kernel_dir.name], ensure_native_kernel=False)
        return kname

    def _safe_shutdown(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        client = getattr(self, "kc", None)
        if client is not None:
            client.stop_channels()
        try:
            self.km.shutdown_kernel(now=False)
        except Exception:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass
        finally:
            if hasattr(self, "_kernel_dir"):
                self._kernel_dir.cleanup()

    def _ask_llm_decision(
        self,
        code: str,
        goal: str,
        output: str,
        run_time: float
    ):
        remaining_time = self.cfg.execution_timeout - run_time
        runtime_minutes = run_time / 60
        max_runtime_minutes = self.cfg.execution_timeout / 60
        remaining_minutes = remaining_time / 60

        lines = output.splitlines()
        if len(lines) > 500:
            output = '\n'.join(lines[:250] + ["... (truncated) ..."] + lines[-250:])

        messages = [
            {
                "role": "system",
                "content": """You are BizMind ML's local experiment execution monitor.
Use observed output and the configured time budget to decide whether to continue.
Do not assume every model prints a loss: a CPU tree model or preprocessing step
can be healthy without frequent output. Do not infer improvement without an actual
validation score. Stop on clear failure or inability to finish within the budget.

Consider these factors:
1. Is the loss exploding (becoming very large or NaN)?
2. Is the loss decreasing normally over time?
3. Are there any error messages indicating failure?
4. Does the output suggest normal training/execution progress?
5. Based on current progress and remaining time, is it possible to complete within the time limit?

Respond in the following format:

<action>
CONTINUE/STOP
</action>

<explanation>
Your rationale for the action. Describe the current progress, your estimated remaining time, and explain why you think the execution should continue or stop. DO NOT GIVE SUGGESTIONS ON BUG FIXES.
</explanation>

."""
            },
            {
                "role": "user",
                "content": f"""
Code being executed:
```python
{code}
```

Goal: {goal}

Runtime Information:
- Current runtime: {runtime_minutes:.2f} minutes ({run_time:.1f} seconds)
- Maximum runtime: {max_runtime_minutes:.2f} minutes ({self.cfg.execution_timeout} seconds)
- Remaining time: {remaining_minutes:.2f} minutes ({remaining_time:.1f} seconds)

Current output:
```
{output}
```

Based on the current progress, runtime information, and output analysis, should the execution continue?
Consider whether the task can realistically be completed within the remaining time.

Respond in the following format:

<action>
CONTINUE/STOP
</action>

<explanation>
Your rationale for the action. Describe the current progress, your estimated remaining time, and explain why you think the execution should continue or stop.
</explanation>
"""
            }
        ]

        response = query_llm(self.cfg.llm, messages, required_fields=["action", "explanation"])

        should_continue = "continue" in response["action"][0].lower()
        explanation = response["explanation"][0] if response["explanation"] else "No explanation provided"

        return should_continue, explanation

    def _collect_iopub_outputs(self, msg_id: str, code: str, goal: str) -> ExecutionResult:
        output, error, success, llm_terminated, timeout = "", "", True, False, False
        start_time = time()
        last_decision_time = start_time
        while True:
            current_time = time()
            if current_time - last_decision_time > self.cfg.execution_inspect_interval:
                last_decision_time = current_time
                should_continue, explanation = self._ask_llm_decision(
                    code, goal, process_backspace_chars(output), current_time - start_time
                )
                if not should_continue:
                    llm_terminated = True
                    error = explanation
                    break

            if current_time - start_time > self.cfg.execution_timeout:
                timeout = True
                break

            try:
                msg = self.kc.get_iopub_msg(timeout=10)
            except Exception:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "status":
                if content.get("execution_state") == "idle":
                    break
            elif msg_type == "stream":
                output += content.get("text", "")
            elif msg_type == "error":
                success = False
                for line in content.get("traceback", []):
                    error += line + "\n"

        if timeout | llm_terminated:
            success = False

        execution_time = time() - start_time
        output, error = process_backspace_chars(output), process_backspace_chars(error)
        return ExecutionResult(
            success=success,
            llm_terminated=llm_terminated,
            timeout=timeout,
            output=output,
            error=error,
            execution_time=execution_time
        )

    def append_cell(self, code: str, goal: str) -> ExecutionResult:
        msg_id = self.kc.execute(code, allow_stdin=False, store_history=True, stop_on_error=True)
        result = self._collect_iopub_outputs(msg_id, code, goal)
        if result.timeout or result.llm_terminated:
            self.km.interrupt_kernel()
        self.cells.append(code=code, success=result.success)
        return result

    def get_notebook_code(self) -> str:
        return self.cells.get_notebook_code()
