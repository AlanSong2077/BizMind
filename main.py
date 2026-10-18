"""Run a local business task through a measured experiment feedback loop."""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from bizmind.config import Config
from bizmind.task_input import load_custom_task, stage_custom_data


def main():
    parser = ArgumentParser(description="BizMind ML — local, self-iterating ML experiments")
    parser.add_argument("--config", type=Path, default=Path("bizmind.yaml"))
    parser.add_argument("--task-id", type=str, help="Local task name used in the run directory")
    parser.add_argument("--task-file", type=Path, help="UTF-8 business task in YAML, Markdown, or text")
    parser.add_argument("--train-data", type=Path, help="Labeled local training file/directory")
    parser.add_argument("--predict-data", type=Path, help="Unlabeled local prediction file/directory")
    parser.add_argument("--data-dir", type=Path, help="Alternative local data directory described by the task")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    for argument, field in (("task_id", "task_id"), ("task_file", "task_input_file"),
                            ("train_data", "task_train_data"), ("predict_data", "task_predict_data"),
                            ("data_dir", "task_input_dir"), ("device", "execution_device")):
        value = getattr(args, argument)
        if value is not None:
            setattr(config, field, value)
    config.__post_init__()
    config.task_description = load_custom_task(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config.agent_workspace_dir = config.agent_workspace_dir / config.task_id / timestamp
    config.task_description = stage_custom_data(config, config.task_description)

    # Keep --help and input validation usable before loading the ML runtime.
    from bizmind.agent import Agent
    Agent(config).run()


if __name__ == "__main__":
    main()
