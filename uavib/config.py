"""Configuration for the UAViB pipeline.

All hyper-parameter defaults match the paper (Section: Implementation Details).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import yaml


@dataclass
class UAViBConfig:
    # --- Coarse pass ---
    coarse_tokens: int = 144           # N_0, ~10-15% of full-resolution budget
    k_passes: int = 4                  # K stochastic passes for agreement
    mc_dropout: bool = True            # use MC-dropout / light TTA for agreement
    free_form_samples: int = 10        # M samples for free-form semantic entropy
    sample_temperature: float = 0.7
    nucleus_p: float = 0.9

    # --- Region grid & attribution ---
    grid_h: int = 8                    # R = grid_h x grid_w
    grid_w: int = 8
    alpha: float = 0.5                 # attention vs. perturbation-sensitivity balance

    # --- Budget ---
    b_min: int = 128                   # B_min
    b_max: int = 1024                  # B_max (native full-resolution budget)
    region_floor: int = 2              # b_min per region (~coarse tokens/region); low-
                                       # uncertainty regions retain coarse resolution so
                                       # the budget concentrates on high-uncertainty ROIs
    tau: float = 0.3                   # target-uncertainty threshold
    gamma: float = 8.0                 # logistic sharpness
    step_tokens: int = 4               # water-filling resolution step

    # --- Progressive refinement ---
    epsilon: float = 0.01              # stop when uncertainty reduction < epsilon
    max_refine_steps: int = 3          # t_max
    top_m_regions: int = 8             # regions re-encoded per refinement step;
                                       # concentrating on the few highest-uncertainty
                                       # regions lets them reach full resolution

    # --- Calibrator ---
    calibrator_hidden: int = 16
    calibrator_lr: float = 1e-3
    calibrator_epochs: int = 200
    calibrator_batch: int = 256
    calibrator_lambda: float = 1.0     # ECE penalty weight
    ece_bins: int = 15

    # --- Misc ---
    seed: int = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "UAViBConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)

    @property
    def num_regions(self) -> int:
        return self.grid_h * self.grid_w
