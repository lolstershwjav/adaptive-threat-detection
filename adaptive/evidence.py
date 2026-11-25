"""Read benchmark evidence from plain or compressed JSON."""

import gzip
import json
from pathlib import Path


def load_study(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding="utf8") as source:
            return json.load(source)
    return json.loads(path.read_text(encoding="utf8"))
