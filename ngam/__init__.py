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
from ngam.dxgi import (
    DXGIAdapterInfo,
    enumerate_adapters,
    get_best_dml_device_id,
    get_hardware_adapters,
)
from ngam.engine import Decider, UniversalDecider, resolve_providers
from ngam.schema import (
    BaseDecision,
    Choice,
    DecisionResult,
    Noul,
    Score,
    build_decision_result,
    compute_normalized_entropy,
    compute_shannon_entropy,
    is_high_entropy,
)

__version__ = "1.1.0"

__all__ = [
    "Decider",
    "UniversalDecider",
    "Choice",
    "Score",
    "Noul",
    "BaseDecision",
    "DecisionResult",
    "build_decision_result",
    "compute_shannon_entropy",
    "compute_normalized_entropy",
    "is_high_entropy",
    "resolve_providers",
    "get_hardware_adapters",
    "get_best_dml_device_id",
    "enumerate_adapters",
    "DXGIAdapterInfo",
    "ensure_model",
    "is_model_available",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_REPO_ID",
]
