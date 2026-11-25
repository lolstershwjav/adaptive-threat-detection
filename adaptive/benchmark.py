"""Run a paired, multi-seed benchmark and export aggregate evidence."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from adaptive.data import synthetic, unsw
from adaptive.replay import POLICIES, replay


def fraction(num, den):
    return float(num / den) if den else None


def assess(scores, alerts, labels):
    malicious = labels != "benign"
    both = malicious.any() and not malicious.all()
    needed = .60
    matched = None
    if both:
        fpr, tpr, _ = roc_curve(malicious, scores, drop_intermediate=False)
        matched = float(fpr[np.flatnonzero(tpr >= needed)[0]])
    by_family = {}
    for family in np.unique(labels[malicious]):
        mask = labels == family
        by_family[family] = {"n": int(mask.sum()), "tp": int(alerts[mask].sum())}
    return {
        "fpr": fraction(alerts[~malicious].sum(), (~malicious).sum()),
        "recall": fraction(alerts[malicious].sum(), malicious.sum()),
        "precision": fraction(alerts[malicious].sum(), alerts.sum()),
        "auc": float(roc_auc_score(malicious, scores)) if both else None,
        "ap": float(average_precision_score(malicious, scores)) if both else None,
        "fpr_at_60_recall": matched,
        "alerts": int(alerts.sum()),
        "family": by_family,
    }


def evaluate(stream, scenario, name, mode, seed, config):
    initial = stream.train, stream.reference, stream.calibration
    labels = [y for _, y in stream.batches]
    outcomes = list(replay(name, POLICIES[mode], seed, initial,
                           (x for x, _ in stream.batches), config["alpha"],
                           config["gate_level"], config["trim_rate"]))
    pooled = assess(np.concatenate([r["scores"] for r in outcomes]),
                    np.concatenate([r["alerts"] for r in outcomes]),
                    np.concatenate(labels))
    accepted_attacks = sum((y[r["retained"]] != "benign").sum() for r, y in zip(outcomes, labels))
    accepted = sum(len(r["retained"]) for r in outcomes)
    updates = [r["batch"] for r in outcomes if r["updated"]]
    alarms = [r["batch"] for r in outcomes if r["trigger"]]
    alarm_delay = None if stream.drift_start is None else next(
        (t - stream.drift_start for t in alarms if t >= stream.drift_start), None
    )
    unknown_hits = [r["batch"] for r, y in zip(outcomes, labels) if r["alerts"][y == "unknown"].any()]
    unknown_start = next((i for i, y in enumerate(labels) if (y == "unknown").any()), None)
    unknown_delay = unknown_hits[0] - unknown_start if unknown_hits else None
    pooled.update(updates=len(updates), retained_contamination=fraction(accepted_attacks, accepted),
                  drift_delay_batches=alarm_delay, unknown_delay_batches=unknown_delay,
                  score_ms=float(np.median([r["score_ms"] for r in outcomes])),
                  update_ms=float(sum(r["update_ms"] for r in outcomes)),
                  initial_fit_ms=outcomes[0]["initial_fit_ms"])
    timeline = []
    for row, y in zip(outcomes, labels):
        scores = row["scores"]
        stats = assess(scores, row["alerts"], y)
        timeline.append({**stats, "batch": row["batch"], "shift": row["shift"],
                         "trigger": row["trigger"], "updated": row["updated"],
                         "threshold": row["threshold"], "median_score": float(np.median(scores)),
                         "p95_score": float(np.quantile(scores, .95)),
                         "p99_score": float(np.quantile(scores, .99)),
                         "true_anomalies": int((y != "benign").sum())})
    return dict(seed=seed, scenario=scenario, model=name, policy=mode,
                metrics=pooled, timeline=timeline, counts=stream.counts)


def mean_interval(values, rng):
    values = np.asarray(values)
    if not len(values):
        return None
    samples = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high), n=len(values))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"], row["model"], row["policy"]].append(row)
    rng = np.random.default_rng(2027)
    summary = []
    for (scenario, model, policy), runs in sorted(groups.items()):
        frozen = {r["seed"]: r["metrics"] for r in groups[scenario, model, "frozen"]}
        metrics = {}
        for field in runs[0]["metrics"]:
            if field == "family":
                continue
            values = [(r["metrics"][field], frozen[r["seed"]][field]) for r in runs]
            metrics[field] = mean_interval([v for v, _ in values if v is not None], rng)
            if field in {"fpr", "recall", "fpr_at_60_recall", "ap", "auc"}:
                metrics[field + "_vs_frozen"] = mean_interval(
                    [v - b for v, b in values if v is not None and b is not None], rng
                )
        families = sorted({family for r in runs for family in r["metrics"]["family"]})
        for family in families:
            counts = [r["metrics"]["family"][family] for r in runs if family in r["metrics"]["family"]]
            metrics["recall_" + family] = mean_interval([m["tp"] / m["n"] for m in counts], rng)
        summary.append(dict(scenario=scenario, model=model, policy=policy, metrics=metrics))
    return summary


def validate(config):
    if config["batch_size"] < 80 or config["steps"] < 16:
        raise ValueError("Study requires at least 80 rows per batch and 16 batches")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Supply distinct seeds")
    if not (0 < config["alpha"] < .5 and 0 <= config["trim_rate"] < 1 and config["gate_level"] > 0):
        raise ValueError("Invalid alpha, trim rate, or gate threshold")
    for scenario, prevalence in config["scenarios"].items():
        if scenario.removesuffix("_burst") not in {"stationary", "gradual", "sudden"} or not 0 <= prevalence <= 1:
            raise ValueError(f"Invalid scenario {scenario}")


def run(config, unsw_paths=None, *, progress=False):
    validate(config)
    rows = []
    specs = [(m, p) for m in config["models"] for p in
             (POLICIES if m in {"iforest", "ensemble"} else ["frozen", "gated_trim"])]
    seeds, size, steps = config["seeds"], config["batch_size"], config["steps"]
    for seed in seeds:
        cases = {k: synthetic(seed, k.removesuffix("_burst"), size, steps, v)
                 for k, v in config["scenarios"].items()}
        if unsw_paths:
            cases["unsw"] = unsw(*unsw_paths, seed, size, steps)
        for scenario, stream in cases.items():
            for model, policy in specs:
                rows.append(evaluate(stream, scenario, model, policy, seed, config))
        if progress:
            print(f"Finished seed {seed} ({len(cases)} scenarios)", file=sys.stderr, flush=True)
    return dict(config=config, environment={"python": platform.python_version(),
                  "numpy": np.__version__, "scikit_learn": sklearn.__version__},
                summary=aggregate(rows), runs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--unsw", type=Path, nargs=2, metavar=("TRAIN", "TEST"))
    parser.add_argument("--output", type=Path, default=Path("results/study.json"))
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.unsw, progress=True)
    if args.unsw:
        result["unsw_sha256"] = [hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                                 for p in args.unsw]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    temp.replace(args.output)


if __name__ == "__main__":
    main()
