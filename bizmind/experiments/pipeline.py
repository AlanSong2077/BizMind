from dataclasses import dataclass
from pathlib import Path

from bizmind.utils.metric import MetricValue


@dataclass
class Pipeline:
    """One completed local experiment, including unsuccessful attempts."""
    id: str
    title: str
    description: str
    code: str
    full_code: str
    metric: MetricValue
    prediction: Path | None
    output_dir: Path | None
    suggestions: str | None

    def __post_init__(self):
        if not self.metric.is_worst and (self.prediction is None or not self.prediction.is_file()):
            raise FileNotFoundError(f"Prediction artifact is missing: {self.prediction}")

    def to_str(self, full_code=False):
        return f"""
<experiment>
<id>{self.id}</id>
<title>{self.title}</title>
<description>{self.description}</description>
<code>{self.full_code if full_code else self.code}</code>
<validation_metric>{self.metric}</validation_metric>
<suggestions>{self.suggestions or ''}</suggestions>
</experiment>
"""
