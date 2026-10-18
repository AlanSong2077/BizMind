"""Load orchestration units while replacing only external service boundaries."""
import __future__
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_classes(filename, names, **dependencies):
    path = ROOT / filename
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(dependencies)
    exec(compile(tree, filename, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace
