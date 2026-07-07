"""UAViB: Uncertainty-Calibrated Adaptive Vision-Token Budgeting.

A plug-and-play, inference-time framework that lets a *frozen* multimodal LLM
decide, from its own calibrated predictive uncertainty, how many vision tokens
to spend and where to spend them.

Public API::

    from uavib import UAViB, UAViBConfig
    from uavib.backends import DummyBackend

    pipeline = UAViB(backend=DummyBackend(), config=UAViBConfig())
    result = pipeline.run(image, question, candidates)
"""

from .config import UAViBConfig
from .types import AnswerOutput, QueryResult, RegionGrid
from .pipeline import UAViB

__all__ = [
    "UAViB",
    "UAViBConfig",
    "AnswerOutput",
    "QueryResult",
    "RegionGrid",
    "__version__",
]

__version__ = "0.1.0"
