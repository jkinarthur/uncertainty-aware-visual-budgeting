"""Unit tests for the UAViB pipeline (run with: pytest)."""

import numpy as np

from uavib import UAViB, UAViBConfig
from uavib.backends import DummyBackend
from uavib.budget import allocate, global_budget
from uavib.calibration import CalibrationHead, expected_calibration_error
from uavib.data import DATASETS, calibration_pool, load_dataset
from uavib.eval import evaluate_all
from uavib.metrics import risk_coverage_auc, auroc_error
from uavib.train import fit_calibrator


def test_budget_monotone_in_uncertainty():
    cfg = UAViBConfig()
    lo = global_budget(0.05, cfg)
    hi = global_budget(0.95, cfg)
    assert cfg.b_min <= lo < hi <= cfg.b_max


def test_allocate_respects_caps_and_budget():
    cfg = UAViBConfig()
    r = cfg.num_regions
    scores = np.full(r, 1.0 / r)
    caps = np.full(r, 20.0)
    b = allocate(256, scores, caps, cfg)
    assert (b <= caps).all()
    assert (b >= cfg.region_floor).all()
    assert abs(b.sum() - 256) <= cfg.step_tokens * 2


def test_calibrator_reduces_ece():
    rng = np.random.default_rng(0)
    n = 2000
    # correctness driven by a latent margin; entropy anti-correlated with it
    latent = rng.normal(size=n)
    correct = (latent + rng.normal(scale=0.5, size=n) > 0).astype(float)
    entropy = np.clip(0.5 - 0.3 * latent + rng.normal(scale=0.1, size=n), 0, 1)
    Z = np.stack([entropy, np.abs(latent), rng.random(n), entropy, 1 - entropy], 1)
    head = CalibrationHead(in_dim=5, hidden=16).fit(Z, correct, epochs=100, verbose=False)
    conf = head.predict_proba(Z)
    raw = 1 - entropy
    assert expected_calibration_error(conf, correct) < expected_calibration_error(raw, correct)


def test_pipeline_runs_and_saves_tokens():
    cfg = UAViBConfig()
    backend = DummyBackend()
    pipe = UAViB(backend, cfg)
    samples = load_dataset("rsvqa-lr", n=20, seed=3)
    r = pipe.run(samples[0].image, samples[0].question, samples[0].candidates, samples[0].answer)
    assert 0 < r.tokens_used <= cfg.b_max
    assert 0.0 <= r.confidence <= 1.0
    assert r.correct in (True, False)


def test_uavib_beats_fixed_on_calibration():
    cfg = UAViBConfig()
    backend = DummyBackend()
    head = fit_calibrator(backend, calibration_pool(n_per_dataset=60), cfg, verbose=False)
    datasets = {n: load_dataset(n, n=80, seed=2) for n in DATASETS}
    res = evaluate_all(backend, datasets, cfg, calibrator=head, progress=False)
    assert res["uavib"]["_all"]["avg_tokens"] < res["full-resolution"]["_all"]["avg_tokens"]
    assert res["uavib"]["_all"]["ece"] < res["fixed-512"]["_all"]["ece"]


def test_metric_bounds():
    conf = np.array([0.9, 0.8, 0.2, 0.6])
    correct = np.array([1, 1, 0, 1])
    assert 0 <= risk_coverage_auc(conf, correct) <= 1
    assert 0 <= auroc_error(1 - conf, correct) <= 1
