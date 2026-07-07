"""Qwen2.5-VL backend (real MLLM). Requires the ``gpu`` extra + a GPU."""

from __future__ import annotations

from typing import List

from .hf_common import HFBackend


class QwenVLBackend(HFBackend):
    name = "qwen2.5-vl"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    patch_size = 28

    def _load(self):  # pragma: no cover - requires GPU + weights
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device,
            attn_implementation="eager",  # needed for output_attentions
        ).eval()

    def _image_token_id(self) -> int:  # pragma: no cover
        return self._model.config.image_token_id

    def _build_inputs(self, images: List, question: str, answer: str = ""):  # pragma: no cover
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if answer:
            text = text + answer
        inputs = self._processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        ).to(self.device)
        n_ans = 1
        if answer:
            n_ans = len(self._processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
        return inputs, max(n_ans, 1)
