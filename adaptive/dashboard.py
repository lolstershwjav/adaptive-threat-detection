"""Assemble an offline dashboard from measured per-batch evidence."""

import argparse
import json
from pathlib import Path

from adaptive.evidence import load_study


def render(source: Path, template: Path, target: Path) -> None:
    results = load_study(source)
    rows = [{k: run[k] for k in ("scenario", "model", "policy", "seed", "metrics", "timeline")}
            for run in results["runs"]]
    content = json.dumps(rows, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.read_text().replace("__REPLAY_DATA__", content), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/study.json"))
    parser.add_argument("--template", type=Path, default=Path("dashboard/template.html"))
    parser.add_argument("--output", type=Path, default=Path("results/replay.html"))
    args = parser.parse_args()
    render(args.source, args.template, args.output)


if __name__ == "__main__":
    main()
