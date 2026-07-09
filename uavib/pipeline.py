"""The UAViB inference pipeline (paper Algorithm: UAViB inference).

Given a frozen backend, a trained (or identity) calibrator, and a query, it:
  1. runs a coarse global pass and computes uncertainty features z;
  2. maps z to a calibrated uncertainty U_hat via the calibration head;
  3. attributes uncertainty to regions (attention + perturbation sensitivity);
  4. sets a global budget B(U_hat) and distributes it across regions;
  5. progressively refines high-uncertainty regions until the marginal
     uncertainty reduction drops below epsilon or B_max / t_max is reached.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from .attribution import attribute
from .budget import allocate, global_budget
from .calibration import CalibrationHead, IdentityCalibrator
from .config import UAViBConfig
from .backends.base import MLLMBackend
from .types import QueryResult
from .uncertainty import compute_features


class UAViB:
    def __init__(
        self,
        backend: MLLMBackend,
        config: Optional[UAViBConfig] = None,
        calibrator=None,
    ):
        self.backend = backend
        self.cfg = config or UAViBConfig()
        self.calibrator = calibrator or IdentityCalibrator()

    def set_calibrator(self, calibrator) -> None:
        self.calibrator = calibrator

    def _calibrated_uncertainty(self, features) -> float:
        z = features.as_vector().reshape(1, -1)
        return float(np.clip(self.calibrator.uncertainty(z)[0], 0.0, 1.0))

    @staticmethod
    def _top_margin(probs) -> float:
        p = np.asarray(probs, dtype=np.float64)
        if p.size < 2:
            return 1.0
        q = np.sort(p)
        return float(q[-1] - q[-2])

    def run(
        self,
        image,
        question: str,
        candidates: Sequence[str],
        answer: Optional[str] = None,
    ) -> QueryResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        candidates = list(candidates)

        # (1) coarse pass + uncertainty features
        coarse = self.backend.coarse_answer(image, question, candidates, cfg.coarse_tokens)
        sampled = self.backend.sample_answers(
            image, question, candidates, cfg.k_passes, cfg.coarse_tokens
        )
        features = compute_features(coarse, sampled)

        # (2) calibrated global uncertainty
        u_global = self._calibrated_uncertainty(features)

        # (3) region attribution
        region_u = attribute(self.backend, image, question, candidates, coarse, cfg)
        caps = self.backend.region_caps(image, cfg.grid_h, cfg.grid_w)

        # (4) global budget + per-region allocation
        budget = global_budget(u_global, cfg)
        region_budgets = allocate(budget, region_u, caps, cfg)

        current = coarse
        best_conf = 1.0 - u_global
        u_prev = u_global
        margin_prev = self._top_margin(coarse.probs)
        u_steps = [u_global]
        tokens_spent = int(coarse.tokens_used)
        history = [{
            "step": 0, "tokens": int(coarse.tokens_used),
            "uncertainty": u_global, "answer": coarse.pred,
        }]

        # (5) progressive refinement -- only for the most uncertain queries.
        # A data-driven gate (top ``refine_fraction`` of calibrated uncertainty,
        # fitted on the calibration pool) decides who gets extra tokens; it falls
        # back to the absolute ``refine_gate`` only when no threshold was fitted.
        # If the frozen model is already confident, extra tokens tend to flip a
        # correct answer for the worse, so we collapse to the coarse answer. This
        # keeps the refine rate controlled and guarantees UAViB never underperforms
        # its own coarse pass on the confident majority.
        gate = getattr(self.calibrator, "refine_threshold", None)
        if gate is None:
            gate = cfg.refine_gate
        steps = 0
        if u_global > gate:
            for step in range(1, cfg.max_refine_steps + 1):
                # expand only the top-m most uncertain regions this step
                order = np.argsort(-region_u)
                active = np.zeros_like(region_budgets)
                keep = order[: cfg.top_m_regions]
                active[keep] = region_budgets[keep]
                # unrefined regions keep coarse-level tokens
                coarse_per_region = cfg.coarse_tokens / cfg.num_regions
                active = np.maximum(active, np.where(active > 0, active, coarse_per_region))

                refined = self.backend.answer_with_budget(
                    image, question, candidates, active, cfg.grid_h, cfg.grid_w
                )
                feat_r = compute_features(refined, sampled)
                u_now = self._calibrated_uncertainty(feat_r)
                u_steps.append(u_now)
                steps = step
                # tokens are spent whether or not we adopt the refined answer
                tokens_spent = max(tokens_spent, int(refined.tokens_used))
                history.append({
                    "step": step, "tokens": int(refined.tokens_used),
                    "uncertainty": u_now, "answer": refined.pred,
                })

                # No-regret acceptance: adopt the refinement only if it actually
                # reduced uncertainty. If the extra crops raised uncertainty, they
                # are misleading the frozen model -- revert to the previous answer
                # and stop refining.
                if u_now > u_prev:
                    break

                # Optional stability guard: refuse refinements that reduce
                # calibrated uncertainty but weaken answer confidence margin.
                margin_now = self._top_margin(refined.probs)
                if margin_now < (margin_prev + cfg.min_margin_gain):
                    break

                current = refined
                features = feat_r
                best_conf = 1.0 - u_now
                margin_prev = margin_now

                # re-attribute and reallocate residual for the next step
                region_u = attribute(self.backend, image, question, candidates, refined, cfg)
                region_budgets = allocate(budget, region_u, caps, cfg)

                if (u_prev - u_now) < cfg.epsilon or refined.tokens_used >= cfg.b_max:
                    break
                u_prev = u_now

        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred = current.pred
        # Selective-prediction signal: expose the coarse-pass difficulty score
        # U_hat by default. U_hat is the model's holistic per-query uncertainty
        # from the global pass and empirically ranks correctness best; refinement
        # inflates confidence and compresses the range, so the trajectory mean
        # (legacy) and the final-pass estimate rank worse. ``report_uncertainty``
        # selects the source ("coarse" | "final" | "mean").
        report_mode = getattr(cfg, "report_uncertainty", "coarse")
        if report_mode == "final":
            rep_unc = float(u_steps[-1])
        elif report_mode == "mean":
            rep_unc = float(np.mean(u_steps))
        else:  # "coarse"
            rep_unc = float(u_global)
        rep_unc = float(np.clip(rep_unc, 0.0, 1.0))
        rep_conf = 1.0 - rep_unc
        result = QueryResult(
            answer=pred,
            confidence=rep_conf,
            uncertainty=rep_unc,
            tokens_used=int(tokens_spent),
            num_refine_steps=steps,
            region_budgets=region_budgets,
            region_uncertainty=region_u,
            features=features,
            latency_ms=latency_ms,
            correct=(None if answer is None else pred == answer),
            history=history,
        )
        return result
