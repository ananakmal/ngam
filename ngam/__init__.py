"""
ngam: Ultra-Fast Typed Neural Decision Engine
=============================================
Probabilistic System-1 routing, continuous ordinal scoring, and epistemic
truth verification across Windows, macOS, and Linux in <5ms.
"""

from ngam.downloader import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_REPO_ID,
    ensure_model,
    is_model_available,
)
from ngam.engine import Decider, UniversalDecider, resolve_providers
from ngam.schema import (
    BaseDecision,
    Choice,
    DecisionResult,
    Noul,
    Score,
    compute_normalized_entropy,
    compute_shannon_entropy,
    is_high_entropy,
)

__version__ = "1.0.0"

__all__ = [
    "Decider",
    "UniversalDecider",
    "Choice",
    "Score",
    "Noul",
    "BaseDecision",
    "DecisionResult",
    "compute_shannon_entropy",
    "compute_normalized_entropy",
    "is_high_entropy",
    "resolve_providers",
    "ensure_model",
    "is_model_available",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_REPO_ID",
]
