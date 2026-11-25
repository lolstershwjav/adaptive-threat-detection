"""Tests for replay isolation, gradient correctness and the benchmark entry point."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from unittest.mock import patch

from adaptive.benchmark import assess, run
from adaptive.data import synthetic
from adaptive.model import Autoencoder, Scorer
from adaptive.replay import MedianDrift, POLICIES, cutoff, replay

ROOT = Path(__file__).resolve().parents[1]


class StudyTests(unittest.TestCase):
    def test_manual_gradients_against_finite_differences(self):
        values = np.random.default_rng(12).normal(size=(9, 7))
        for variational in (False, True):
            with self.subTest(variational=variational):
                model = Autoencoder(13, variational, epochs=0)
                model.fit(values)
                state = copy.deepcopy(model.rng.bit_generator.state)
                manual = model._gradients(values)

                def loss():
                    model.rng.bit_generator.state = copy.deepcopy(state)
                    _, _, _, decoded, aux = model._forward(values, True)
                    reconstruction = ((values - decoded) ** 2).mean()
                    if aux is not None:
                        mean, _, logvar, _, _ = aux
                        reconstruction += .01 * np.mean(mean ** 2 + np.expm1(logvar) - logvar)
                    return reconstruction

                eps = 1e-5
                for params, grad in zip(model.params, manual):
                    index = (0,) * params.ndim
                    original = params[index]
                    params[index] = original + eps
                    plus = loss()
                    params[index] = original - eps
                    minus = loss()
                    params[index] = original
                    self.assertAlmostEqual(grad[index], (plus - minus) / (2 * eps), places=7)

    def test_all_models_score_extreme_input(self):
        r = synthetic(12, "stationary", 90, 17, .05)
        extremes = np.vstack([r.reference[:4], np.full((1, 12), 1e7)])
        for name in ("iforest", "lof", "ocsvm", "ae", "vae", "ensemble"):
            model = Scorer(name, 1)
            model.fit(r.train, r.reference)
            result = model.score(extremes)
            self.assertEqual(result.shape, (5,))
            self.assertTrue(np.isfinite(result).all())

    def test_replay_scores_before_any_current_batch_refit(self):
        r = synthetic(19, "sudden", 90, 16, .05)
        warmup = r.train, r.reference, r.calibration
        features = [x for x, _ in r.batches]
        with patch.object(Scorer, "fit", autospec=True, side_effect=Scorer.fit) as fit:
            def first(mode):
                return next(replay("iforest", POLICIES[mode], 4, warmup, features, .05, .32, .15))

            baseline = first("frozen")
            rolling = first("rolling_trim")
            call = fit.call_args.args
            self.assertEqual(fit.call_count, 3)
            np.testing.assert_array_equal(call[2], features[0][2::4])
            self.assertTrue((rolling["retained"] % 4 < 2).all())
        np.testing.assert_array_equal(baseline["scores"], rolling["scores"])
        np.testing.assert_array_equal(baseline["alerts"], rolling["alerts"])
        self.assertTrue(rolling["updated"])

    def test_gated_replay_requires_two_consecutive_batches(self):
        r = synthetic(19, "sudden", 90, 16, .05)
        warmup = r.train, r.reference, r.calibration
        features = [r.reference, r.reference + 3, r.reference, r.reference + 3, r.reference + 3]
        rows = list(replay("iforest", POLICIES["gated_trim"], 4, warmup, features, .05, .32, .15))
        self.assertEqual([r["batch"] for r in rows if r["updated"]], [4])
        self.assertEqual(MedianDrift(r.reference).distance(r.reference), 0)

    def test_metrics_for_rank_ties_and_single_class(self):
        self.assertEqual(cutoff(np.array([0, 1, 1, 3]), .25), 3)
        scores = np.array([2., 2., 0.])
        labels = np.array(["unknown", "benign", "benign"])
        measured = assess(scores, scores > 1, labels)
        self.assertEqual(measured["fpr_at_60_recall"], .5)
        self.assertEqual(measured["recall"], 1.)
        self.assertIsNone(assess(scores, scores > 1, np.full(3, "benign"))["auc"])

    def test_cli_emits_paired_reproducible_metrics(self):
        config = json.loads((ROOT / "configs/study.json").read_text())
        config.update(seeds=[45], scenarios={"sudden": .05}, models=["ocsvm"],
                      batch_size=80, steps=16)
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "study.json"
            out = Path(tmp) / "out.json"
            config_path.write_text(json.dumps(config))
            subprocess.run([sys.executable, "-m", "adaptive.benchmark", "--config", str(config_path),
                            "--output", str(out)], cwd=ROOT, check=True,
                           env={**os.environ, "OPENBLAS_NUM_THREADS": "1"}, capture_output=True)
            first = json.loads(out.read_text())
        again = run(config)
        self.assertEqual([r["timeline"] for r in first["runs"]],
                         [r["timeline"] for r in again["runs"]])
        self.assertEqual(len(first["summary"]), 2)


if __name__ == "__main__":
    unittest.main()
