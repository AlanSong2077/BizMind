"""Detect usable training hardware without requiring NVIDIA tools on every host."""
from dataclasses import dataclass
import os
import platform


@dataclass(frozen=True)
class HardwareInfo:
    device: str
    device_names: tuple[str, ...] = ()
    memory_gb: tuple[float | None, ...] = ()
    # Physical tokens, in the same order as the parent's logical CUDA devices.
    cuda_visible_devices: tuple[str, ...] = ()
    cpu_count: int = 1
    system: str = ""

    def to_prompt(self) -> str:
        details = [f"Local execution platform: {self.system}; CPU logical cores: {self.cpu_count}.",
                   f"Selected training device: {self.device}."]
        if self.device == "cuda":
            for index, (name, memory) in enumerate(zip(self.device_names, self.memory_gb)):
                capacity = f"{memory:.1f} GiB" if memory is not None else "unknown memory capacity"
                details.append(f"CUDA device {index}: {name}, {capacity}.")
            details.append("Use only the CUDA devices visible to the notebook; do not change CUDA_VISIBLE_DEVICES.")
        elif self.device == "mps":
            details.extend([
                f"Apple Metal GPU: {self.device_names[0]}; memory is shared with the host.",
                "Use torch.device('mps') for supported PyTorch operations; use CPU for libraries or operations without MPS support.",
                "CUDA, NVIDIA tools, NCCL and CUDA-specific packages are unavailable on this device.",
            ])
            if self.memory_gb and self.memory_gb[0] is not None:
                details.append(f"Recommended Metal working-set limit: {self.memory_gb[0]:.1f} GiB "
                               "of shared memory (not reserved or dedicated VRAM).")
        else:
            details.append("Use CPU training. Do not require CUDA, NVIDIA tools, or an MPS accelerator.")
        details.append("In notebook code, os.environ['BIZMIND_DEVICE'] identifies the selected backend. "
                       "Respect CPU limits and scale batch/model sizes to available memory; no GPU memory reservation is provided.")
        return "\n".join(details)


def _visible_cuda_tokens(count: int) -> tuple[str, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return tuple(str(index) for index in range(count))
    tokens = []
    for token in visible.split(","):
        token = token.strip()
        # CUDA stops enumerating at an invalid or duplicate entry. Never fall
        # back to physical device 0 when the caller explicitly hides devices.
        if not (token.isdigit() or token.startswith(("GPU-", "MIG-"))) or token in tokens:
            break
        tokens.append(token)
    return tuple(tokens[:count])


def detect_hardware(device: str = "auto", max_gpu_count: int = 1) -> HardwareInfo:
    """Prefer usable CUDA, then macOS MPS, then CPU; honor a GPU count cap.

    Explicit unavailable accelerators fail clearly. ``max_gpu_count=0`` always
    selects CPU. PyTorch is optional for CPU-only workflows.
    """
    device = device.lower()
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("execution.device must be auto, cuda, mps, or cpu")
    if max_gpu_count < 0:
        raise ValueError("execution.max_gpu_count must be nonnegative")
    common = dict(cpu_count=os.cpu_count() or 1, system=platform.system())
    cpu = HardwareInfo(device="cpu", **common)
    if device == "cpu" or max_gpu_count == 0:
        return cpu
    try:
        import torch
    except (ImportError, OSError):
        if device != "auto":
            raise RuntimeError(f"Requested {device} device is unavailable: PyTorch is not installed or cannot load") from None
        return cpu

    if device in {"auto", "cuda"}:
        try:
            if torch.cuda.is_available():
                tokens = _visible_cuda_tokens(torch.cuda.device_count())[:max_gpu_count]
                if tokens:
                    names, memory = [], []
                    for index in range(len(tokens)):
                        try:
                            properties = torch.cuda.get_device_properties(index)
                            names.append(properties.name)
                            memory.append(properties.total_memory / 1024 ** 3)
                        except Exception:
                            names.append(f"CUDA device {index}")
                            memory.append(None)
                    return HardwareInfo(device="cuda", device_names=tuple(names), memory_gb=tuple(memory),
                                        cuda_visible_devices=tokens, **common)
        except Exception:
            # Driver problems should not break automatic CPU/MPS operation.
            pass
        if device == "cuda":
            raise RuntimeError("Requested cuda device is unavailable or hidden by CUDA_VISIBLE_DEVICES")

    if device in {"auto", "mps"} and common["system"] == "Darwin":
        try:
            if torch.backends.mps.is_available():
                try:
                    recommended_memory = torch.mps.recommended_max_memory() / 1024 ** 3
                except (AttributeError, RuntimeError):
                    recommended_memory = None
                return HardwareInfo(device="mps", device_names=("Apple Metal (MPS)",),
                                    memory_gb=(recommended_memory,), **common)
        except (AttributeError, RuntimeError):
            pass
    if device == "mps":
        raise RuntimeError("Requested mps device is unavailable; it requires a supported Mac and PyTorch MPS build")
    return cpu
