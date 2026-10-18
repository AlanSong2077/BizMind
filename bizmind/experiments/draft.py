from dataclasses import dataclass


@dataclass
class Draft:
    id: str
    title: str
    description: str
    codebase: str | None
    codebase_content: str | None
    code: str
