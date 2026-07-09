"""Domain-agnostic calibration head (paper Section: Domain-Agnostic Calibration
Head + Algorithm: Calibrator training).

The only learnable component in UAViB. It maps the 5-dim, domain-invariant,
logit-derived feature vector z to a calibrated confidence p_hat = g_phi(z) that
the coarse answer is correct; uncertainty is U_hat = 1 - p_hat.

Implemented as a small two-layer MLP with a temperature-scaled logit output,
trained by minimising NLL plus a soft-binned ECE penalty (Eq. cal-loss). Pure
NumPy so the core package needs no deep-learning framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-9


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Standard binned ECE."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def _soft_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Differentiable-ish soft-binned ECE used as a training penalty."""
    centers = (np.linspace(0, 1, n_bins + 1)[:-1] + 0.5 / n_bins)
    width = 1.0 / n_bins
    # soft (triangular) bin membership
    w = np.maximum(0.0, 1.0 - np.abs(conf[:, None] - centers[None, :]) / width)
    w_sum = w.sum(axis=0) + _EPS
    acc_bin = (w * correct[:, None]).sum(axis=0) / w_sum
    conf_bin = (w * conf[:, None]).sum(axis=0) / w_sum
    weight = w_sum / w_sum.sum()
    return float(np.sum(weight * (acc_bin - conf_bin) ** 2))


@dataclass
class _Params:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    log_temp: float


class CalibrationHead:
    """Two-layer MLP + temperature scaling, NumPy Adam training."""

    def __init__(self, in_dim: int = 5, hidden: int = 16, seed: int = 0):
        self.in_dim = in_dim
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.p = _Params(
            W1=rng.normal(0, scale1, (in_dim, hidden)),
            b1=np.zeros(hidden),
            W2=rng.normal(0, scale2, (hidden, 1)),
            b2=np.zeros(1),
            log_temp=0.0,
        )
        self.mean = np.zeros(in_dim)
        self.std = np.ones(in_dim)
        self.refine_threshold = None   # data-driven top-R uncertainty gate (set at fit)
        self._fitted = False

    # --- forward ---
    def _standardize(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.mean) / (self.std + _EPS)

    def _logits(self, Zs: np.ndarray):
        h_pre = Zs @ self.p.W1 + self.p.b1
        h = np.maximum(0.0, h_pre)          # ReLU
        logit = (h @ self.p.W2 + self.p.b2).ravel()
        temp = np.exp(self.p.log_temp)
        return logit / temp, (h_pre, h, logit, temp)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        """Calibrated confidence p_hat = P(correct | z)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        Zs = self._standardize(Z)
        scaled, _ = self._logits(Zs)
        return _sigmoid(scaled)

    def uncertainty(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 - self.predict_proba(Z)

    # --- training ---
    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 200,
        batch: int = 256,
        lam: float = 1.0,
        n_bins: int = 15,
        seed: int = 0,
        verbose: bool = False,
    ) -> "CalibrationHead":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mean, self.std = Z.mean(0), Z.std(0)
        Zs = self._standardize(Z)
        n = len(Zs)
        rng = np.random.default_rng(seed)

        params = [self.p.W1, self.p.b1, self.p.W2, self.p.b2,
                  np.array([self.p.log_temp])]
        m = [np.zeros_like(x) for x in params]
        v = [np.zeros_like(x) for x in params]
        b1a, b2a, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch):
                sel = idx[start:start + batch]
                Zb, yb = Zs[sel], y[sel]
                grads, loss = self._grads(Zb, yb, lam, n_bins)
                t += 1
                for i, g in enumerate(grads):
                    m[i] = b1a * m[i] + (1 - b1a) * g
                    v[i] = b2a * v[i] + (1 - b2a) * (g * g)
                    mhat = m[i] / (1 - b1a ** t)
                    vhat = v[i] / (1 - b2a ** t)
                    params[i] -= lr * mhat / (np.sqrt(vhat) + eps)
                self.p.W1, self.p.b1, self.p.W2, self.p.b2 = params[0], params[1], params[2], params[3]
                self.p.log_temp = float(params[4][0])
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                conf = self.predict_proba(Z)
                print(f"  epoch {epoch:3d} loss={loss:.4f} "
                      f"ece={expected_calibration_error(conf, y, n_bins):.4f}")
        self._fitted = True
        return self

    def _grads(self, Zs, yb, lam, n_bins):
        h_pre = Zs @ self.p.W1 + self.p.b1
        h = np.maximum(0.0, h_pre)
        logit = (h @ self.p.W2 + self.p.b2).ravel()
        temp = np.exp(self.p.log_temp)
        scaled = logit / temp
        p = _sigmoid(scaled)
        nb = len(yb)

        # NLL (BCE) gradient wrt scaled logit
        d_scaled = (p - yb) / nb
        # soft-ECE penalty gradient (finite-difference-free approximation via p)
        soft = _soft_ece(p, yb, n_bins)
        # pull confidence toward empirical accuracy (variance-style surrogate)
        d_scaled = d_scaled + lam * 2.0 * (p * (1 - p)) * (p - yb) / nb * soft

        loss = float(-np.mean(yb * np.log(p + _EPS) + (1 - yb) * np.log(1 - p + _EPS)) + lam * soft)

        d_logit = d_scaled / temp
        d_log_temp = float(np.sum(d_scaled * (-logit / temp)))  # d(logit/temp)/d(log_temp)

        dW2 = h.T @ d_logit[:, None]
        db2 = np.array([d_logit.sum()])
        d_h = d_logit[:, None] @ self.p.W2.T
        d_hpre = d_h * (h_pre > 0)
        dW1 = Zs.T @ d_hpre
        db1 = d_hpre.sum(0)
        return [dW1, db1, dW2, db2, np.array([d_log_temp])], loss

    # --- persistence ---
    def save(self, path: str) -> None:
        obj = {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "W1": self.p.W1.tolist(),
            "b1": self.p.b1.tolist(),
            "W2": self.p.W2.tolist(),
            "b2": self.p.b2.tolist(),
            "log_temp": self.p.log_temp,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "refine_threshold": (None if self.refine_threshold is None
                                 else float(self.refine_threshold)),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    @classmethod
    def load(cls, path: str) -> "CalibrationHead":
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        head = cls(in_dim=obj["in_dim"], hidden=obj["hidden"])
        head.p = _Params(
            W1=np.array(obj["W1"]), b1=np.array(obj["b1"]),
            W2=np.array(obj["W2"]), b2=np.array(obj["b2"]),
            log_temp=float(obj["log_temp"]),
        )
        head.mean = np.array(obj["mean"])
        head.std = np.array(obj["std"])
        rt = obj.get("refine_threshold", None)
        head.refine_threshold = (None if rt is None else float(rt))
        head._fitted = True
        return head


class IdentityCalibrator:
    """Fallback used before a calibrator is trained: confidence = 1 - entropy."""

    refine_threshold = None

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return np.clip(1.0 - Z[:, 0], 0.0, 1.0)  # feature 0 is normalized entropy

    def uncertainty(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 - self.predict_proba(Z)
