"""Configuration for local business tasks and their experiment budgets."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time

import yaml


@dataclass
class LLMConfig:
    model: str = "deepseek/deepseek-flash"
    params: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class Config:
    task_id: str = "business-task"
    task_input_file: Path | None = Path("examples/business_task.yaml")
    task_train_data: Path | None = None
    task_predict_data: Path | None = None
    task_input_dir: Path | None = None
    task_description: str | None = None

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent_num_code_agents: int = 1
    agent_num_iterations: int = 3
    agent_num_iterations_code_agents: int = 20
    agent_time_limit_code_agents: int = 18000
    agent_workspace_dir: Path = Path("workspace")
    agent_prediction_file_name: str = "predictions.csv"

    execution_inspect_interval: int = 600
    execution_timeout: int = 3600
    execution_max_cpu_cores: int = 4
    execution_device: str = "auto"
    execution_max_gpu_count: int = 1
    start_time: float = field(default_factory=time.time, init=False)

    def __post_init__(self):
        for name in ("task_input_file", "task_train_data", "task_predict_data",
                     "task_input_dir", "agent_workspace_dir"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value).expanduser())
        for name in ("task_id", "agent_prediction_file_name"):
            value = getattr(self, name)
            if (not isinstance(value, str) or not value.strip() or value in {".", ".."}
                    or "/" in value or chr(92) in value):
                raise ValueError(f"{name} must be a nonempty file/directory name without path separators.")
        for name in ("agent_num_code_agents", "agent_num_iterations", "agent_num_iterations_code_agents",
                     "agent_time_limit_code_agents", "execution_inspect_interval", "execution_timeout",
                     "execution_max_cpu_cores"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (isinstance(self.execution_max_gpu_count, bool)
                or not isinstance(self.execution_max_gpu_count, int) or self.execution_max_gpu_count < 0):
            raise ValueError("execution_max_gpu_count must be a nonnegative integer.")
        if self.execution_device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("execution_device must be auto, cuda, mps, or cpu.")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        with open(yaml_path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if document is None:
            document = {}
        sections = {"task", "llm", "agent", "execution"}
        if not isinstance(document, dict) or set(document) - sections:
            raise ValueError("Config supports only task, llm, agent, and execution sections; use bizmind.yaml.")
        values = {}
        for section, options in document.items():
            if not isinstance(options, dict):
                raise ValueError(f"{section} must be a mapping.")
            if section == "llm":
                continue
            for key, value in options.items():
                name = f"{section}_{key}"
                if name not in cls.__dataclass_fields__:
                    raise ValueError(f"Unknown configuration option: {section}.{key}")
                values[name] = value
        model = dict(document.get("llm", {}))
        values["llm"] = LLMConfig(
            model=model.pop("model", "deepseek/deepseek-flash"), max_retries=model.pop("max_retries", 3), params=model)
        return cls(**values)

    def to_yaml(self, yaml_path: Path) -> None:
        document = {section: {} for section in ("task", "agent", "execution")}
        for name in self.__dataclass_fields__:
            if name in {"llm", "start_time", "task_description"}:
                continue
            section, key = name.split("_", 1)
            value = getattr(self, name)
            document[section][key] = str(value) if isinstance(value, Path) else value
        document["llm"] = {**self.llm.params, "model": self.llm.model, "max_retries": self.llm.max_retries}
        with open(yaml_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
