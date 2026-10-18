"""Load local business tasks and assemble data without changing source files."""
from pathlib import Path

import yaml


def _load_reference(value, directory: Path, field: str) -> str | None:
    """Resolve optional reference text without importing or executing supplied code."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if (not isinstance(value, dict) or set(value) != {"path"}
            or not isinstance(value["path"], str) or not value["path"].strip()):
        raise ValueError(f"{field} must be text, null, or a mapping with a nonempty path.")
    source = (directory / Path(value["path"]).expanduser()).resolve()
    if not source.exists():
        raise FileNotFoundError(f"{field} file not found: {source}")
    if not source.is_file():
        raise ValueError(f"{field}.path must point to a UTF-8 text file: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field}.path must point to a UTF-8 text file: {source}") from error
    if "\x00" in text:
        raise ValueError(f"{field}.path must point to a UTF-8 text file: {source}")
    return text if text.strip() else None


def load_custom_task(cfg) -> str:
    if cfg.task_input_file is None:
        raise ValueError("A business task requires task.input_file or --task-file.")
    task_file = Path(cfg.task_input_file).expanduser().resolve()
    text = task_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Task description file is empty: {task_file}")
    if task_file.suffix.lower() not in {".yaml", ".yml"}:
        return text

    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("A YAML task file must contain a mapping of business requirements.")
    data = document.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("The task file's data section must be a mapping.")
    for field, role in (("task_train_data", "train"), ("task_predict_data", "predict")):
        # Explicit CLI/config paths take precedence over paths in the task file.
        if getattr(cfg, field, None) is None and data.get(role) is not None:
            if not isinstance(data[role], str) or not data[role].strip():
                raise ValueError(f"data.{role} must be a nonempty local path.")
            source = Path(data[role]).expanduser()
            setattr(cfg, field, (task_file.parent / source).resolve())

    # Do not expose original source paths in the training agents' task prompt.
    business = {key: value for key, value in document.items() if key != "data"}
    metadata = {key: value for key, value in data.items() if key not in {"train", "predict"}}
    if metadata:
        business["data"] = metadata
    if not business:
        raise ValueError("The task file must include business requirements, not only data paths.")
    for field in ("existing_solution", "existing_code"):
        if field in business:
            business[field] = _load_reference(business[field], task_file.parent, field)
    return yaml.safe_dump(business, allow_unicode=True, sort_keys=False)


def stage_custom_data(cfg, description: str) -> str:
    sources = [getattr(cfg, "task_train_data", None), getattr(cfg, "task_predict_data", None)]
    if all(source is None for source in sources):
        if cfg.task_input_dir is None:
            raise ValueError("Provide both task.train_data and task.predict_data, or --data-dir.")
        source_dir = Path(cfg.task_input_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Local task data directory not found: {source_dir}")
        if cfg.agent_workspace_dir.resolve().is_relative_to(source_dir):
            raise ValueError("The task workspace must not be inside an input data directory.")
        cfg.task_input_dir = source_dir
        return description
    if any(source is None for source in sources):
        raise ValueError("Provide both task.train_data and task.predict_data.")

    sources = [Path(source).expanduser().resolve() for source in sources]
    destination = (cfg.agent_workspace_dir / "task_data").resolve()
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Local task data not found: {source}")
        if not (source.is_file() or source.is_dir()):
            raise ValueError(f"Local task data must be a file or directory: {source}")
        if source.is_dir() and destination.is_relative_to(source):
            raise ValueError("The task workspace must not be inside an input data directory.")
    if sources[0] == sources[1]:
        raise ValueError("Training and prediction data must be different sources.")
    if any(source.is_dir() and other.is_relative_to(source)
           for source, other in ((sources[0], sources[1]), (sources[1], sources[0]))):
        raise ValueError("Training and prediction data must not overlap or contain each other.")

    destination.mkdir(parents=True, exist_ok=False)
    relative_paths = []
    for role, source in zip(("train_data", "predict_data"), sources):
        target = destination / role
        if source.is_dir():
            target.symlink_to(source, target_is_directory=True)
        else:
            target.mkdir()
            target = target / source.name
            target.symlink_to(source)
        relative_paths.append(target.relative_to(destination).as_posix())
    cfg.task_input_dir = destination
    cfg.task_train_data, cfg.task_predict_data = sources

    return description + f"""

## Runtime dataset mapping (relative to the active task input directory)
- Labeled training data: {relative_paths[0]}
- Unlabeled prediction data: {relative_paths[1]}
These paths override source paths or example filenames in the business description.
In a coding notebook, read from ../input/data/ using these relative
paths. This points to the evaluator's public split, not the
original training source. Preserve prediction row identifiers and output format.
Do not modify source files; extract archives and write generated files in the
working/public directories. Never read or copy hidden validation labels.
"""
