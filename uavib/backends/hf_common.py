"""Shared logic for real Hugging Face MLLM backends (Qwen2.5-VL, LLaVA-NeXT).

Design (matches the paper's "look closer only when unsure"):

  * The model always receives a low-resolution *thumbnail* of the whole image
    (the coarse pass, ~``coarse_tokens`` vision tokens).
  * Refinement spends extra tokens by appending *high-resolution crops* of the
    most uncertain regions as additional image tiles — natively supported by
    both Qwen2.5-VL and LLaVA-NeXT (AnyRes / multi-image inputs).
  * ``tokens_used`` = thumbnail tokens + sum of crop tokens, so cost tracks the
    per-region budget exactly.

Closed-form answers are scored by length-normalised teacher-forced
log-likelihood over the candidate set; answer-to-vision attention is read
best-effort from ``output_attentions`` and pooled into the region grid.

Everything torch/transformers is imported lazily so the core package stays
dependency-free.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..types import AnswerOutput
from .base import MLLMBackend


def _lazy_torch():
    import torch  # noqa: WPS433
    return torch


def crop_region(pil_image, idx: int, grid_h: int, grid_w: int):
    """Crop region ``idx`` (row-major) from a PIL image."""
    w, h = pil_image.size
    r, c = divmod(idx, grid_w)
    left = int(c * w / grid_w)
    upper = int(r * h / grid_h)
    right = int((c + 1) * w / grid_w)
    lower = int((r + 1) * h / grid_h)
    return pil_image.crop((left, upper, right, lower))


def resize_to_tokens(pil_image, n_tokens: int, patch: int = 28):
    """Resize a tile so its patch grid yields about ``n_tokens`` tokens."""
    from PIL import Image  # noqa: WPS433

    n_tokens = max(int(n_tokens), 1)
    side_patches = max(1, int(round(math.sqrt(n_tokens))))
    side_px = side_patches * patch
    return pil_image.resize((side_px, side_px), Image.BILINEAR)


class HFBackend(MLLMBackend):
    """Base class; subclasses set ``model_id`` and load the right classes."""

    name = "hf"
    model_id: str = ""
    patch_size: int = 28

    def __init__(self, model_id: Optional[str] = None, device: str = "cuda",
                 dtype: str = "bfloat16", last_layer_frac: float = 0.25):
        self.model_id = model_id or self.model_id
        self.device = device
        self.dtype = dtype
        self.last_layer_frac = last_layer_frac
        self._model = None
        self._processor = None

    # --- loading (subclass overrides _load) ---
    def _ensure_loaded(self):
        if self._model is None:
            self._load()

    def _load(self):  # pragma: no cover - requires GPU + weights
        raise NotImplementedError

    def _image_token_id(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def _build_inputs(self, images: List, question: str, answer: str = ""):
        """Build model inputs from a list of PIL images + a chat prompt.

        Subclasses implement the model-specific chat template. Returns a dict of
        tensors on ``self.device`` plus the number of answer tokens appended.
        """
        raise NotImplementedError  # pragma: no cover

    # --- candidate scoring ---
    def _score_candidates(self, images, question, candidates) -> np.ndarray:
        torch = _lazy_torch()
        self._ensure_loaded()
        logprobs = []
        for cand in candidates:
            inputs, n_ans = self._build_inputs(images, question, cand)
            with torch.no_grad():
                out = self._model(**inputs)
            logits = out.logits[0]
            labels = inputs["input_ids"][0]
            # score only the answer tokens (last n_ans positions)
            lp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = labels[1:]
            tok_lp = lp[torch.arange(len(tgt)), tgt]
            ans_lp = tok_lp[-n_ans:].mean()  # length-normalised
            logprobs.append(float(ans_lp))
        logprobs = np.array(logprobs, dtype=np.float64)
        probs = np.exp(logprobs - logprobs.max())
        return probs / probs.sum()

    def _region_attention(self, images, question, candidate, grid_h, grid_w) -> Optional[np.ndarray]:
        torch = _lazy_torch()
        try:
            inputs, _ = self._build_inputs(images, question, candidate)
            with torch.no_grad():
                out = self._model(**inputs, output_attentions=True)
            attns = out.attentions  # tuple(num_layers)[B, heads, q, k]
            n_layers = len(attns)
            start = int(n_layers * (1 - self.last_layer_frac))
            sel = torch.stack(attns[start:]).mean(0).mean(1)[0]  # [q, k]
            ids = inputs["input_ids"][0]
            vis_mask = (ids == self._image_token_id())
            if vis_mask.sum() == 0:
                return None
            ans_rows = sel[-1]  # attention from last position
            vis_attn = ans_rows[vis_mask].float().cpu().numpy()
            # pool the (thumbnail) vision tokens into the region grid
            return self._pool_to_grid(vis_attn, grid_h, grid_w)
        except Exception:  # pragma: no cover - attention is best-effort
            return None

    @staticmethod
    def _pool_to_grid(vis_attn: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
        n = len(vis_attn)
        side = max(1, int(round(math.sqrt(n))))
        grid = np.zeros(grid_h * grid_w, dtype=np.float64)
        for p in range(min(n, side * side)):
            pr, pc = divmod(p, side)
            gr = min(grid_h - 1, int(pr * grid_h / side))
            gc = min(grid_w - 1, int(pc * grid_w / side))
            grid[gr * grid_w + gc] += vis_attn[p]
        s = grid.sum()
        return grid / s if s > 0 else np.full_like(grid, 1.0 / len(grid))

    # --- MLLMBackend interface ---
    def coarse_answer(self, image, question, candidates, coarse_tokens):
        thumb = resize_to_tokens(image, coarse_tokens, self.patch_size)
        probs = self._score_candidates([thumb], question, candidates)
        pred = candidates[int(np.argmax(probs))]
        attn = self._region_attention([thumb], question, pred, 8, 8)
        return AnswerOutput(list(candidates), probs, attn, int(coarse_tokens))

    def answer_with_budget(self, image, question, candidates, region_budgets,
                           grid_h, grid_w):
        region_budgets = np.asarray(region_budgets)
        thumb_tokens = int(min(region_budgets.sum(),
                               max(64, region_budgets.sum() * 0.3)))
        images = [resize_to_tokens(image, thumb_tokens, self.patch_size)]
        used = thumb_tokens
        floor = 4
        for i, b in enumerate(region_budgets):
            if b > floor * 2:
                tile = crop_region(image, i, grid_h, grid_w)
                images.append(resize_to_tokens(tile, int(b), self.patch_size))
                used += int(b)
        probs = self._score_candidates(images, question, candidates)
        pred = candidates[int(np.argmax(probs))]
        attn = self._region_attention(images[:1], question, pred, grid_h, grid_w)
        return AnswerOutput(list(candidates), probs, attn, int(used))

    def answer_with_occlusions(self, image, question, candidates,
                               grid_h, grid_w, coarse_tokens):
        from PIL import Image  # noqa: WPS433

        outs = []
        base = resize_to_tokens(image, coarse_tokens, self.patch_size)
        arr = np.array(base)
        H, W = arr.shape[:2]
        mean_color = arr.reshape(-1, arr.shape[-1]).mean(0).astype(arr.dtype)
        for i in range(grid_h * grid_w):
            r, c = divmod(i, grid_w)
            occ = arr.copy()
            occ[int(r * H / grid_h):int((r + 1) * H / grid_h),
                int(c * W / grid_w):int((c + 1) * W / grid_w)] = mean_color
            probs = self._score_candidates([Image.fromarray(occ)], question, candidates)
            outs.append(AnswerOutput(list(candidates), probs, None, int(coarse_tokens)))
        return outs

    def sample_answers(self, image, question, candidates, k, coarse_tokens):
        # Mild test-time augmentation (resolution jitter) to surface epistemic
        # uncertainty via cross-pass disagreement. These passes feed ONLY the
        # agreement feature, not the reported answer. Perturbations are kept
        # deliberately MILD and semantics-preserving: under a confident model a
        # flip only occurs on genuinely borderline queries, making disagreement
        # a rare but high-precision error signal. (Aggressive perturbation adds
        # false disagreements and destroys this correlation.)
        answers = []
        for j in range(k):
            jitter = int(coarse_tokens * (0.85 + 0.3 * (j / max(k - 1, 1))))
            thumb = resize_to_tokens(image, jitter, self.patch_size)
            probs = self._score_candidates([thumb], question, candidates)
            answers.append(candidates[int(np.argmax(probs))])
        return answers

    def region_caps(self, image, grid_h, grid_w):
        w, h = image.size
        r = grid_h * grid_w
        # tokens a region yields at native resolution
        tile_w, tile_h = w / grid_w, h / grid_h
        cap = (tile_w / self.patch_size) * (tile_h / self.patch_size)
        return np.full(r, max(4.0, cap), dtype=np.float64)
