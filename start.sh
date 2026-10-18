#!/usr/bin/env bash
# Usage: ./start.sh [main.py arguments]
# Override the interpreter with BIZMIND_PYTHON=/path/to/python.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ -n "${BIZMIND_PYTHON:-}" ]]; then
    python_bin="$BIZMIND_PYTHON"
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
    python_bin="$project_dir/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
else
    python_bin="python3"
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python interpreter not found: %s\nSet BIZMIND_PYTHON to a Python 3.11+ executable.\n' "$python_bin" >&2
    exit 127
fi

if ! "$python_bin" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    printf 'BizMind requires Python 3.11 or newer.\n' >&2
    exit 1
fi

if ! "$python_bin" -c 'import yaml' >/dev/null 2>&1; then
    printf 'Missing dependency: PyYAML. Install project dependencies with:\n  "%s" -m pip install -r requirements.txt\n' "$python_bin" >&2
    exit 1
fi

exec "$python_bin" "$project_dir/main.py" "$@"
