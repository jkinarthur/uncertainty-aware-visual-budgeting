"""MLLM backends for UAViB.

``DummyBackend`` runs anywhere (no deps). The real backends are imported lazily
so importing this package never requires torch/transformers.
"""

from .base import MLLMBackend
from .dummy import DummyBackend, DummyImage

__all__ = ["MLLMBackend", "DummyBackend", "DummyImage", "get_backend"]


def get_backend(name: str, **kwargs) -> MLLMBackend:
    """Factory: 'dummy' | 'qwen' | 'llava'."""
    name = name.lower()
    if name in ("dummy", "sim"):
        return DummyBackend(**kwargs)
    if name in ("qwen", "qwen2.5-vl", "qwenvl"):
        from .qwen import QwenVLBackend
        return QwenVLBackend(**kwargs)
    if name in ("llava", "llava-next", "llavanext"):
        from .llava import LlavaNextBackend
        return LlavaNextBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}")
