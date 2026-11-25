"""Score-before-update replay, calibration and drift policy ablations."""

from dataclasses import dataclass
import time

import numpy as np

from adaptive.model import Scorer


@dataclass(frozen=True)
class Policy:
    refit: bool = False
    threshold: bool = False
    gate: bool = False
    trim: bool = False


POLICIES = {
    "frozen": Policy(),
    "threshold": Policy(threshold=True),
    "rolling": Policy(refit=True, threshold=True),
    "rolling_trim": Policy(refit=True, threshold=True, trim=True),
    "gated": Policy(refit=True, threshold=True, gate=True),
    "gated_trim": Policy(refit=True, threshold=True, gate=True, trim=True),
    "gated_fit_only": Policy(refit=True, gate=True, trim=True),
}


def cutoff(scores, alpha):
    return float(np.quantile(scores, 1 - alpha, method="higher"))


class MedianDrift:
    """Compare the incoming batch median against the reference in robust scale units."""

    def __init__(self, reference: np.ndarray):
        self.update(reference)

    def update(self, reference):
        lower, upper = np.quantile(reference, [.25, .75], axis=0)
        self.center = np.median(reference, axis=0)
        self.width = np.maximum(upper - lower, 1e-5)

    def distance(self, batch):
        shifts = np.abs(np.median(batch, axis=0) - self.center) / self.width
        return float(np.median(shifts))


def replay(name, policy, seed, initial, batches, alpha, gate_level, trim_rate):
    train, reference, calibration = initial
    history = train.copy()
    scorer = Scorer(name, seed)
    clock = time.perf_counter
    start = clock()
    scorer.fit(train, reference)
    boundary = cutoff(scorer.score(calibration), alpha)
    fit_ms = (clock() - start) * 1000
    drift = MedianDrift(reference)
    consecutive = 0
    for step, values in enumerate(batches):
        start = clock()
        scores = scorer.score(values)
        decision = scores > boundary
        score_ms = (clock() - start) * 1000
        current_boundary = boundary
        distance = drift.distance(values)
        consecutive = consecutive + 1 if distance >= gate_level else 0
        trigger = consecutive >= 2
        updated = False
        retained = np.empty(0, dtype=int)
        update_start = clock()
        if (policy.refit or policy.threshold) and (trigger or not policy.gate):
            # Interleave roles so burst position does not select the role.
            indices = np.arange(len(values))
            eligible = indices[indices % 4 < 2]
            if policy.trim:
                # Fixed rank budget permits adaptation even after a large benign shift.
                keep = max(1, int(len(eligible) * (1 - trim_rate)))
                eligible = np.sort(eligible[np.argsort(scores[eligible], kind="stable")[:keep]])
            if policy.refit:
                retained = eligible
                history = np.concatenate([history, values[retained]])[-len(train):]
                reference_rows = values[indices % 4 == 2]
                scorer = Scorer(name, seed)
                scorer.fit(history, reference_rows)
                drift.update(reference_rows)
                consecutive = 0
            if policy.threshold:
                boundary = cutoff(scorer.score(values[indices % 4 == 3]), alpha)
            updated = True
        yield {
            "batch": step, "scores": scores, "alerts": decision, "threshold": current_boundary,
            "shift": distance, "trigger": trigger, "updated": updated, "retained": retained,
            "score_ms": score_ms, "update_ms": (clock() - update_start) * 1000,
            "initial_fit_ms": fit_ms,
        }
