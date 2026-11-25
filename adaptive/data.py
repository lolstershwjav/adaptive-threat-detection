"""Public benchmark inputs and controlled telemetry replay."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
            "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "smean", "dmean")


@dataclass(frozen=True)
class Replay:
    train: np.ndarray
    reference: np.ndarray
    calibration: np.ndarray
    batches: list[tuple[np.ndarray, np.ndarray]]
    drift_start: int | None
    counts: dict


def synthetic(seed: int, regime: str, size: int, steps: int, contamination: float) -> Replay:
    rng = np.random.default_rng(seed)
    features = 12
    loadings = rng.normal(0, .48, (3, features))

    def base(n):
        return rng.normal(size=(n, 3)) @ loadings + rng.normal(0, .65, (n, features))

    train, ref, cal = base(size * 3), base(size), base(size)
    batches = []
    drift_start = 4 if regime != "stationary" else None
    observed = np.array(["benign", "volume", "asymmetry", "unknown"], dtype="U12")
    for step in range(steps):
        # Generate all draws in all regimes so comparisons share rows when possible.
        x = base(size)
        rank = rng.random(size)
        family = rng.integers(1, 4, size=size)
        shape = rng.choice([-1, 1], size=size)
        progress = max(0., min(1., (step - 3) / 7))
        shift = {"stationary": 0., "gradual": progress, "sudden": float(step >= 4)}[regime]
        x[:, :10] += 1.65 * shift
        x[:, 5:] *= 1 + .25 * shift
        load = contamination if 8 <= step < 12 else .05
        family = np.where(rank < load, family, 0)
        family = np.where((family == 3) & (step < steps // 2), 2, family)
        x[family == 1, :3] += 2.6
        x[family == 2, 3:6] -= 2.4
        x[family == 3, 8:] += 2.5 * shape[family == 3, None]
        batches.append((x, observed[family]))
    return Replay(train, ref, cal, batches, drift_start, {})


def unsw(train_path: Path, test_path: Path, seed: int, size: int, steps: int) -> Replay:
    """Use the official split sizes and a stratiform replay with preserved within-bin order.

    Published partition CSVs lack timestamps, so batch order is only a constructed replay.
    Balanced spreading of rows across source-order bins avoids evaluating a single label block.
    """
    columns = [*FEATURES, "label", "attack_cat"]
    train = pd.read_csv(train_path, usecols=columns).loc[lambda frame: frame.label == 0, list(FEATURES)]
    test = pd.read_csv(test_path, usecols=columns).loc[:, columns]
    # Equality is defined in the actual model feature space, independent of labels.
    train = train.loc[np.isfinite(train.to_numpy()).all(axis=1)].drop_duplicates()
    test = test.loc[np.isfinite(test.loc[:, list(FEATURES)].to_numpy()).all(axis=1)]
    duplicated = test.duplicated(subset=list(FEATURES), keep=False)
    test = test.loc[~duplicated].copy()
    seen = pd.MultiIndex.from_frame(train).get_indexer(pd.MultiIndex.from_frame(test.loc[:, list(FEATURES)]))
    test = test.loc[seen < 0].reset_index(drop=True)
    if len(train) < 5 * size or len(test) < steps * size:
        raise ValueError("Not enough distinct finite feature rows for the configured replay")
    rng = np.random.default_rng(seed)
    x = np.log1p(train.iloc[rng.permutation(len(train))[:size * 5]].to_numpy())
    labels = np.where(test.label.eq(0), "benign", test.attack_cat.astype(str).str.strip().str.lower())
    xs = np.log1p(test.loc[:, list(FEATURES)].to_numpy())
    batches = []
    for part in np.array_split(np.arange(len(test)), steps):
        selected = np.sort(rng.choice(part, size, replace=False))
        batches.append((xs[selected], labels[selected]))
    return Replay(x[:size * 3], x[size * 3:size * 4], x[size * 4:], batches, None,
                  {"distinct_benign_train": len(train), "distinct_test": len(test),
                   "duplicate_test_rows_excluded": int(duplicated.sum())})
