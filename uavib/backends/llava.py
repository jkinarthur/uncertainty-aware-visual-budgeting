"""LLaVA-NeXT backend (real MLLM). Requires the ``gpu`` extra + a GPU."""

from __future__ import annotations

from typing import List

from .hf_common import HFBackend


class LlavaNextBackend(HFBackend):
    name = "llava-next"
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    patch_size = 14

    def _load(self):  # pragma: no cover - requires GPU + weights
        import torch
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device,
            attn_implementation="eager",
        ).eval()

    def _image_token_id(self) -> int:  # pragma: no cover
        return self._model.config.image_token_index

    def _build_inputs(self, images: List, question: str, answer: str = ""):  # pragma: no cover
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        if answer:
            text = text + answer
        inputs = self._processor(
            images=images, text=text, return_tensors="pt"
        ).to(self.device)
        n_ans = 1
        if answer:
            n_ans = len(self._processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
        return inputs, max(n_ans, 1)
