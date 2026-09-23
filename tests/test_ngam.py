"""
tests.test_ngam
===============
Comprehensive unit, mock, integration, and contract test suite for ngam.
Verifies 22 distinct invariants across provider resolution, Shannon entropy,
Pydantic contracts, offline caching, mock inference, real model e2e, CORS, and daemon.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import platform
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import onnxruntime as ort
import pytest

from ngam import (
    BaseDecision,
    Choice,
    Decider,
    DecisionResult,
    Noul,
    Score,
    UniversalDecider,
    compute_normalized_entropy,
    compute_shannon_entropy,
    ensure_model,
    is_high_entropy,
    is_model_available,
    resolve_providers,
)
from ngam.downloader import DEFAULT_CACHE_DIR, DEFAULT_MODEL_DIR


# ============================================================================
# 1. Cross-Platform Provider Resolution Tests (Tests 1-5)
# ============================================================================

def test_resolve_providers_darwin():
    """Test 1: Verify macOS resolves CoreMLExecutionProvider before CPU fallback."""
    with patch("platform.system", return_value="Darwin"):
        with patch("onnxruntime.get_available_providers", return_value=["CoreMLExecutionProvider", "CPUExecutionProvider"]):
            providers = resolve_providers()
            assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_windows():
    """Test 2: Verify Windows prioritizes DmlExecutionProvider before CPU fallback."""
    with patch("platform.system", return_value="Windows"):
        with patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            providers = resolve_providers()
            assert providers == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_linux():
    """Test 3: Verify Linux prioritizes CUDAExecutionProvider and OpenVINOExecutionProvider."""
    with patch("platform.system", return_value="Linux"):
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]):
            providers = resolve_providers()
            assert providers == ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_override():
    """Test 4: Verify explicit preferred provider override takes top priority."""
    with patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
        providers = resolve_providers(preferred="cpu")
        assert providers[0] == "CPUExecutionProvider"


def test_resolve_providers_env_var():
    """Test 5: Verify NGAM_PROVIDER environment variable override."""
    with patch.dict(os.environ, {"NGAM_PROVIDER": "cpu"}):
        with patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            providers = resolve_providers()
            assert providers[0] == "CPUExecutionProvider"


# ============================================================================
# 2. Shannon Entropy & Ambiguity Gating Tests (Tests 6-8)
# ============================================================================

def test_shannon_entropy_deterministic():
    """Test 6: A completely deterministic distribution must have zero entropy."""
    probs = [1.0, 0.0, 0.0]
    h = compute_shannon_entropy(probs)
    assert pytest.approx(h, abs=1e-5) == 0.0
    h_norm = compute_normalized_entropy(probs)
    assert pytest.approx(h_norm, abs=1e-5) == 0.0

    ambiguous, reason = is_high_entropy(probs)
    assert ambiguous is False
    assert reason is None


def test_shannon_entropy_uniform():
    """Test 7: Uniform distributions must reach theoretical maximum entropy log2(k)."""
    # 2 options: H = 1.0 bit
    probs_2 = [0.5, 0.5]
    assert pytest.approx(compute_shannon_entropy(probs_2), abs=1e-5) == 1.0
    assert pytest.approx(compute_normalized_entropy(probs_2), abs=1e-5) == 1.0

    # 4 options: H = 2.0 bits
    probs_4 = [0.25, 0.25, 0.25, 0.25]
    assert pytest.approx(compute_shannon_entropy(probs_4), abs=1e-5) == 2.0
    assert pytest.approx(compute_normalized_entropy(probs_4), abs=1e-5) == 1.0


def test_is_high_entropy_detection():
    """Test 8: Verify gating flags for near-uniform distributions and narrow separations."""
    # Near-uniform distribution -> flagged ambiguous
    near_uniform = [0.34, 0.33, 0.33]
    ambiguous, reason = is_high_entropy(near_uniform, threshold_ratio=0.85)
    assert ambiguous is True
    assert reason is not None
    assert "High Shannon entropy" in reason

    # Clear winner -> NOT ambiguous
    clear_dist = [0.95, 0.03, 0.02]
    ambiguous, reason = is_high_entropy(clear_dist, threshold_ratio=0.85)
    assert ambiguous is False
    assert reason is None

    # Narrow margin with low top candidate -> ambiguous
    tied_low = [0.41, 0.40, 0.19]
    ambiguous, reason = is_high_entropy(tied_low, threshold_ratio=0.99, min_margin=0.05)
    assert ambiguous is True
    assert "Narrow candidate separation" in reason


# ============================================================================
# 3. Pydantic V2 Schema Contracts Tests (Tests 9-14)
# ============================================================================

def test_choice_schema_valid():
    """Test 9: Verify Choice[T] typing, validation, and probability dictionary."""
    choice = Choice[str](
        decision="billing",
        probabilities={"billing": 0.92, "tech": 0.05, "sales": 0.03},
        options=["billing", "tech", "sales"],
        confidence=0.85,
        entropy=0.35,
        ambiguous=False,
    )
    assert choice.type == "choice"
    assert choice.decision == "billing"
    assert choice.probabilities["billing"] == 0.92
    assert choice.confidence == 0.85
    assert choice.ambiguous is False


def test_choice_schema_invalid_probabilities():
    """Test 10: Verify Choice rejects probability distributions not summing to ~1.0."""
    with pytest.raises(ValueError):
        Choice[str](
            decision="billing",
            probabilities={"billing": 0.20, "tech": 0.10},  # sums to 0.30
            confidence=0.5,
            entropy=0.5,
        )


def test_score_parameterized_bounds():
    """Test 11: Verify Score[Min, Max] syntax and bounded class behavior."""
    ScoreType = Score[1, 10]
    instance = ScoreType(
        score=7.85,
        probabilities={"7": 0.4, "8": 0.6},
        confidence=0.75,
        entropy=0.6,
    )
    assert instance.type == "score"
    assert instance.score == 7.85
    assert instance.min_val == 1
    assert instance.max_val == 10


def test_score_invalid_bounds():
    """Test 12: Verify Score rejects min >= max."""
    with pytest.raises(ValueError):
        _ = Score[10, 5]


def test_noul_schema():
    """Test 13: Verify Noul epistemic verification schema and boolean threshold."""
    noul_true = Noul(
        noul=0.88,
        decision=True,
        probabilities={"false": 0.12, "true": 0.88},
        confidence=0.65,
        entropy=0.54,
    )
    assert noul_true.type == "noul"
    assert noul_true.noul == 0.88
    assert noul_true.decision is True

    noul_false = Noul(
        noul=0.15,
        decision=False,
        probabilities={"false": 0.85, "true": 0.15},
        confidence=0.61,
        entropy=0.61,
    )
    assert noul_false.decision is False


def test_decision_result_serialization():
    """Test 14: Verify DecisionResult round-trip JSON serialization."""
    choice = Choice[str](
        decision="sales",
        probabilities={"sales": 0.90, "support": 0.10},
        confidence=0.80,
        entropy=0.46,
    )
    res = DecisionResult(
        prompt="Enterprise pricing inquiry",
        decision=choice,
        latency_ms=4.2,
        provider="CPUExecutionProvider",
        model_name="ngam-onnx",
        tokens_used=5,
    )
    json_str = res.model_dump_json()
    parsed = DecisionResult.model_validate_json(json_str)
    assert parsed.prompt == res.prompt
    assert parsed.decision.decision == "sales"
    assert parsed.latency_ms == 4.2


# ============================================================================
# 4. Model Downloader & Offline Cache Tests (Tests 15-16)
# ============================================================================

def test_downloader_cached_offline():
    """Test 15: Verify offline mode succeeds when local or dev cache assets are present."""
    assert is_model_available(DEFAULT_MODEL_DIR) is True
    validated_path = ensure_model(model_dir=DEFAULT_MODEL_DIR, offline_mode=True)
    assert os.path.isdir(validated_path)


def test_downloader_offline_failure_on_missing():
    """Test 16: Verify offline mode raises FileNotFoundError when required assets are absent."""
    with pytest.raises(FileNotFoundError):
        ensure_model(model_dir="c:/dev/ngam/non_existent_model_dir_9999", offline_mode=True)


# ============================================================================
# 5. Deterministic Mock Session Inference Tests (Tests 17-20)
# ============================================================================

class MockInferenceSession:
    """Mock ONNX session returning deterministic logits for hermetic testing."""
    def __init__(self, logits: np.ndarray, act_logits: Optional[np.ndarray] = None):
        self._logits = logits
        self._act_logits = act_logits or np.array([[0.9, 0.1]])

    def run(self, output_names, input_feed, run_options=None):
        k = input_feed["marker_pos"].shape[1]
        out_logits = self._logits[:, :k]
        return [out_logits, self._act_logits]

    def get_providers(self):
        return ["CPUExecutionProvider"]


class MockTokenizer:
    """Mock Tokenizer converting text to token IDs."""
    def __init__(self):
        self.mapping = {
            "[CLS]": 50281,
            "[SEP]": 50282,
            "[PAD]": 50283,
            "[MASK]": 50284,
        }

    def token_to_id(self, token: str) -> int:
        return self.mapping.get(token, 100)

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = text.split()
        ids = [self.token_to_id(t) for t in tokens]
        res = MagicMock()
        res.ids = ids
        return res


def test_mock_decide_choice():
    """Test 17: Verify deterministic choice scoring using mock ONNX session."""
    mock_logits = np.array([[10.0, 1.0, -2.0]])
    mock_sess = MockInferenceSession(logits=mock_logits)
    mock_tok = MockTokenizer()

    decider = UniversalDecider(
        custom_session=mock_sess,
        custom_tokenizer=mock_tok,
    )

    result = decider.decide_choice(
        prompt="Sample ticket",
        options=["first_opt", "second_opt", "third_opt"],
    )

    assert result.decision == "first_opt"
    assert result.probabilities["first_opt"] > 0.90
    assert result.confidence > 0.70
    assert result.ambiguous is False


def test_mock_decide_score():
    """Test 18: Verify deterministic ordinal scoring using mock ONNX session."""
    mock_logits = np.array([[-5.0, -5.0, -5.0, -5.0, 10.0]])
    mock_sess = MockInferenceSession(logits=mock_logits)
    mock_tok = MockTokenizer()

    decider = UniversalDecider(
        custom_session=mock_sess,
        custom_tokenizer=mock_tok,
    )

    score_res = decider.decide_score(
        prompt="Critical incident",
        min_val=0,
        max_val=4,
    )

    assert pytest.approx(score_res.score, abs=0.1) == 4.0
    assert score_res.probabilities["4"] > 0.90


def test_mock_decide_noul():
    """Test 19: Verify deterministic noul epistemic verification using mock session."""
    mock_logits = np.array([[-5.0, 5.0]])
    mock_sess = MockInferenceSession(logits=mock_logits)
    mock_tok = MockTokenizer()

    decider = UniversalDecider(
        custom_session=mock_sess,
        custom_tokenizer=mock_tok,
    )

    noul_res = decider.decide_noul(
        prompt="State is operational",
        statement="System is operating normally",
    )

    assert noul_res.decision is True
    assert noul_res.noul > 0.90


def test_out_of_domain_rejection():
    """Test 20: Verify that high-entropy uniform predictions trigger OOD rejection."""
    mock_logits = np.array([[1.0, 1.0, 1.0]])
    mock_sess = MockInferenceSession(logits=mock_logits)
    mock_tok = MockTokenizer()

    decider = UniversalDecider(
        custom_session=mock_sess,
        custom_tokenizer=mock_tok,
    )

    res = decider.decide(
        "asdf qwer zxcv 1234 invalid domain garbage input",
        choice=["billing", "technical", "sales"],
    )

    assert res.is_ambiguous is True
    assert res.is_out_of_domain is True
    assert res.rejection_reason is not None
    assert res.decision.is_out_of_domain is True


# ============================================================================
# 6. Daemon Server & CORS Preflight Tests (Test 21)
# ============================================================================

def test_daemon_server_health_cors_and_decide():
    """Test 21: Verify daemon /healthz, OPTIONS 204 CORS headers, /v1/decide, and shutdown."""
    from ngam.daemon import create_server

    decider = UniversalDecider(preferred_provider="cpu")
    port = 8049
    server = create_server("127.0.0.1", port, decider=decider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.4)

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        # 1. Check /healthz
        conn.request("GET", "/healthz")
        res = conn.getresponse()
        assert res.status == 200
        health = json.loads(res.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["warm"] is True

        # 2. Check OPTIONS CORS preflight
        conn.request("OPTIONS", "/v1/decide")
        opt_res = conn.getresponse()
        assert opt_res.status == 204
        assert opt_res.getheader("Access-Control-Allow-Origin") == "*"
        assert "POST" in opt_res.getheader("Access-Control-Allow-Methods")
        opt_res.read()

        # 3. Check POST /v1/decide
        payload = json.dumps({
            "prompt": "I was charged twice on my credit card. Please refund.",
            "choice": ["billing", "tech", "sales"],
        })
        conn.request("POST", "/v1/decide", body=payload, headers={"Content-Type": "application/json"})
        d_res = conn.getresponse()
        assert d_res.status == 200
        dec_data = json.loads(d_res.read().decode("utf-8"))
        assert dec_data["decision"]["decision"] == "billing"
        assert dec_data["decision"]["type"] == "choice"

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# ============================================================================
# 7. Real Model E2E Spectrum Tests (Test 22)
# ============================================================================

def test_real_model_e2e_full_spectrum():
    """Test 22: Full end-to-end inference evaluating Choice, Score, and Noul routing."""
    decider = Decider(preferred_provider="cpu")

    # 1. Choice: refund routing
    c_res = decider.decide(
        "I was charged twice on my invoice. Please refund the duplicate transaction.",
        choice=["billing", "technical", "sales"],
    )
    assert c_res.decision.decision == "billing"
    assert c_res.decision.probabilities["billing"] > 0.65
    assert c_res.latency_ms < 2500.0

    # 2. Score: severity rating
    s_res = decider.decide(
        "Critical power loss and catastrophic database failure across all clusters.",
        score="0-5",
    )
    assert isinstance(s_res.decision, Score)
    assert 0.0 <= s_res.decision.score <= 5.0

    # 3. Noul: epistemic truth verification
    n_res = decider.decide(
        "The customer requests a full refund for their duplicate billing payment.",
        noul="The customer wants their money refunded.",
    )
    assert isinstance(n_res.decision, Noul)
    assert n_res.decision.decision is True
    assert n_res.decision.noul >= 0.5
