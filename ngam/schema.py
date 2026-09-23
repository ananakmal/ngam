"""
ngam.schema
===========
Pydantic V2 data contracts, Shannon entropy calculations, and epistemic
uncertainty gating for the ngam Universal Neural Decision Engine.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


def compute_shannon_entropy(probabilities: Union[Sequence[float], np.ndarray], base: float = 2.0) -> float:
    """
    Compute Shannon entropy H(P) = -sum(p * log_base(p)) for probability distribution P.
    
    Args:
        probabilities: Sequence or array of probabilities (normalized to ~1.0).
        base: Logarithm base (default 2.0 for bits).
        
    Returns:
        Entropy in bits (>= 0.0).
    """
    if probabilities is None:
        return 0.0
    
    probs = [float(p) for p in probabilities]
    if len(probs) <= 1:
        return 0.0
    
    total = sum(probs)
    if total <= 0:
        return 0.0
    
    # Normalize probabilities to avoid floating precision drift
    norm_probs = [p / total for p in probs]
    entropy = 0.0
    log_base = math.log(base)
    
    for p in norm_probs:
        if p > 1e-12:
            entropy -= p * (math.log(p) / log_base)
            
    return max(0.0, float(entropy))


def compute_normalized_entropy(probabilities: Union[Sequence[float], np.ndarray]) -> float:
    """
    Compute normalized Shannon entropy H_norm(P) = H(P) / log2(k), where k = len(P).
    
    Returns a value in [0.0, 1.0], where 1.0 is maximum uncertainty (uniform distribution).
    """
    if probabilities is None:
        return 0.0
    probs = [float(p) for p in probabilities]
    k = len(probs)
    if k <= 1:
        return 0.0
    h = compute_shannon_entropy(probs, base=2.0)
    h_max = math.log2(k)
    return max(0.0, min(1.0, float(h / h_max))) if h_max > 0 else 0.0


def is_high_entropy(
    probabilities: Union[Sequence[float], np.ndarray],
    threshold_ratio: float = 0.85,
    min_margin: float = 0.05
) -> Tuple[bool, Optional[str]]:
    """
    Determine if a probability distribution is high-entropy (ambiguous / Out-Of-Distribution).
    
    Args:
        probabilities: Sequence or array of probabilities.
        threshold_ratio: Normalized entropy cutoff (default 0.85).
        min_margin: Minimum probability gap between top 2 candidates (default 0.05).
        
    Returns:
        (is_ambiguous, reason_description)
    """
    if probabilities is None:
        return False, None
    probs = [float(p) for p in probabilities]
    k = len(probs)
    if k <= 1:
        return False, None
    
    h_norm = compute_normalized_entropy(probs)
    sorted_probs = sorted(probs, reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if k > 1 else 0.0
    margin = top1 - top2
    
    if h_norm >= threshold_ratio:
        return (
            True,
            f"High Shannon entropy (H_norm={h_norm:.3f} >= {threshold_ratio:.2f}); distribution is near-uniform."
        )
    
    if k >= 2 and margin < min_margin and top1 < (1.5 / k):
        return (
            True,
            f"Narrow candidate separation (top1={top1:.3f}, top2={top2:.3f}, margin={margin:.3f} < {min_margin:.2f}); decision is ambiguous."
        )
        
    return False, None


class BaseDecision(BaseModel):
    """Base contract for all ngam decision outputs."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(..., description="Decision type indicator: 'choice', 'score', or 'noul'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Epistemic confidence metric in [0.0, 1.0]")
    entropy: float = Field(..., ge=0.0, description="Shannon entropy H(P) in bits")
    ambiguous: bool = Field(default=False, description="Flag indicating high entropy or out-of-distribution ambiguity")
    ambiguity_reason: Optional[str] = Field(default=None, description="Detailed rationale if ambiguous is True")
    is_out_of_domain: bool = Field(default=False, description="Flag indicating if input is out-of-domain or rejected due to high entropy")
    rejection_reason: Optional[str] = Field(default=None, description="Rejection rationale if out-of-domain")
    act_probability: Optional[float] = Field(default=None, description="Escalation / act head probability if produced")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Inference latency in milliseconds")
    provider: str = Field(default="CPUExecutionProvider", description="ONNX Runtime execution provider utilized")


T = TypeVar("T")


class Choice(BaseDecision, Generic[T]):
    """Categorical choice decision among a finite set of candidates."""
    type: Literal["choice"] = "choice"
    decision: T = Field(..., description="The winning choice candidate")
    probabilities: Dict[str, float] = Field(..., description="Calibrated softmax probabilities per candidate")
    options: List[str] = Field(default_factory=list, description="Original list of option candidates considered")

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("Probabilities dictionary cannot be empty.")
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Probabilities must sum to approximately 1.0 (got {total:.4f}).")
        return v


class Score(BaseDecision):
    """Ordinal / continuous score decision bounded by [min_val, max_val]."""
    type: Literal["score"] = "score"
    score: float = Field(..., description="Continuous or expected score value")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Distribution across score levels")
    min_val: int = Field(default=0, description="Minimum score bound")
    max_val: int = Field(default=5, description="Maximum score bound")
    legend: Optional[Dict[str, str]] = Field(default=None, description="Semantic description of each score level")

    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        """Enables Score[Min, Max] syntax e.g. Score[1, 10] or Score[0, 5]."""
        if isinstance(item, tuple) and len(item) == 2:
            min_v, max_v = item
            if not (isinstance(min_v, int) and isinstance(max_v, int)):
                raise TypeError(f"Score bounds must be integers, got ({min_v!r}, {max_v!r})")
            if min_v >= max_v:
                raise ValueError(f"Min bound ({min_v}) must be strictly less than Max bound ({max_v})")
            
            subclass_name = f"Score_{min_v}_{max_v}"
            class ParametrizedScore(cls):
                min_val: int = Field(default=min_v, description="Minimum score bound")
                max_val: int = Field(default=max_v, description="Maximum score bound")
                __doc__: str = f"Score model constrained to bounds [{min_v}, {max_v}]."
                
            ParametrizedScore.__name__ = subclass_name
            ParametrizedScore.__qualname__ = subclass_name
            return ParametrizedScore
        return super().__class_getitem__(item)


class Noul(BaseDecision):
    """Binary epistemic verification / truth probability (p[1] holds)."""
    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability that the statement holds")
    decision: bool = Field(..., description="Boolean verdict: True if noul >= 0.5, else False")
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution: {'false': 1.0 - noul, 'true': noul}"
    )


class DecisionResult(BaseModel):
    """Top-level unified response envelope for ngam decision requests."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    prompt: str = Field(..., description="Input query or context state")
    decision: Union[Choice[Any], Score, Noul, Any] = Field(..., description="Primary decision output")
    all_decisions: Optional[Dict[str, Union[Choice[Any], Score, Noul]]] = Field(
        default=None,
        description="Map of decisions if multiple questions were evaluated in a single forward pass"
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Total forward pass and decoding latency in ms")
    provider: str = Field(default="CPUExecutionProvider", description="ONNX execution provider utilized")
    model_name: str = Field(default="ngam-onnx", description="Model architecture identifier")
    tokens_used: int = Field(default=0, ge=0, description="Total sequence tokens processed")
    is_ambiguous: bool = Field(default=False, description="Flag indicating if the primary decision is ambiguous")
    is_out_of_domain: bool = Field(default=False, description="Whether the decision was flagged as out-of-domain")
    rejection_reason: Optional[str] = Field(default=None, description="Rejection rationale if out-of-domain")
