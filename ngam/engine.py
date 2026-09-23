"""
ngam.engine
===========
Universal ONNX Decision Engine powering probabilistic routing, ordinal scoring,
and epistemic verification across Windows (DirectML), macOS (Core ML), and Linux (CUDA).
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from ngam.downloader import DEFAULT_MODEL_DIR, ensure_model, is_model_available
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

logger = logging.getLogger("ngam.engine")


def resolve_providers(preferred: Optional[str] = None) -> List[str]:
    """
    Probe available ONNX Runtime execution providers and return an optimized priority list.
    Prioritizes platform-native hardware acceleration:
      - Darwin (macOS): CoreMLExecutionProvider, CPUExecutionProvider
      - Windows: DmlExecutionProvider, CPUExecutionProvider
      - Linux: CUDAExecutionProvider, OpenVINOExecutionProvider, CPUExecutionProvider
    """
    available = ort.get_available_providers()
    providers: List[str] = []

    provider_aliases: Dict[str, str] = {
        "dml": "DmlExecutionProvider",
        "directml": "DmlExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "cpu": "CPUExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "dmlexecutionprovider": "DmlExecutionProvider",
        "coremlexecutionprovider": "CoreMLExecutionProvider",
        "cudaexecutionprovider": "CUDAExecutionProvider",
        "openvinoexecutionprovider": "OpenVINOExecutionProvider",
        "cpuexecutionprovider": "CPUExecutionProvider",
    }

    env_preferred = os.environ.get("NGAM_PROVIDER") or os.environ.get("AGY_DECIDE_PROVIDER")
    target_preferred = preferred or env_preferred

    if target_preferred:
        matched = provider_aliases.get(target_preferred.lower(), target_preferred)
        if matched in available:
            providers.append(matched)
        else:
            logger.warning(f"Preferred provider '{target_preferred}' not found in available: {available}")

    # Platform-specific native hardware priority list
    system = platform.system().lower()
    if "darwin" in system:
        platform_order = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    elif "windows" in system:
        platform_order = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:  # Linux / other Unix
        platform_order = [
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]

    for candidate in platform_order:
        if candidate in available and candidate not in providers:
            providers.append(candidate)

    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    return providers


class UniversalDecider:
    """
    Universal High-Performance Neural Decision Engine wrapping Laya ONNX runtime.
    Supports single or multi-head evaluation:
      - Choice[T]: Categorical routing & multi-class selection
      - Score[Min, Max]: Calibrated ordinal rating & continuous evaluation
      - Noul: Epistemic truth verification (binary holds probability)
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        offline_mode: bool = False,
        custom_session: Optional[ort.InferenceSession] = None,
        custom_tokenizer: Optional[Tokenizer] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Universal Decision Engine.
        
        Args:
            model_dir: Directory containing model weights (defaults to ~/.cache/ngam/models/laya_onnx)
            preferred_provider: Explicit provider override ('dml', 'coreml', 'cuda', 'cpu')
            offline_mode: Disallow Hugging Face downloads if assets are missing
            custom_session: Optional pre-initialized ONNX session (useful for mocks/tests)
            custom_tokenizer: Optional pre-initialized Tokenizer
            custom_config: Optional rl_agent_config dict
        """
        self.preferred_provider = preferred_provider
        self.offline_mode = offline_mode
        self.providers = resolve_providers(preferred_provider)

        if custom_session is not None:
            self.session = custom_session
            self.model_dir = model_dir or "custom"
            self.config = custom_config or {
                "max_len": 512,
                "head_max_len": 192,
                "temperature": [1.63, 1.25, 1.98],
                "temperature_by_options": {
                    "choice:2": 1.90,
                    "choice:3-5": 1.76,
                    "choice:6-10": 1.00,
                    "score:3-5": 1.25,
                    "noul:2": 1.98,
                },
            }
            self.tokenizer = custom_tokenizer
            self.active_provider = (
                self.session.get_providers()[0] if hasattr(self.session, "get_providers") else "CPUExecutionProvider"
            )
        else:
            self.model_dir = ensure_model(model_dir=model_dir, offline_mode=offline_mode)
            self._load_config()
            self._load_tokenizer()

        self.model_name = self.config.get("model_name", "laya-onnx")
        # Cache special tokens BEFORE initializing session so pre-flight verification can use them
        if self.tokenizer is not None:
            self.cls_id = self.tokenizer.token_to_id("[CLS]")
            self.sep_id = self.tokenizer.token_to_id("[SEP]")
            self.mask_id = self.tokenizer.token_to_id("[MASK]")
            self.pad_id = self.tokenizer.token_to_id("[PAD]")
        else:
            self.cls_id = 50281
            self.sep_id = 50282
            self.mask_id = 50284
            self.pad_id = 50283

        if custom_session is None:
            self._init_session()

    def _load_config(self) -> None:
        cfg_path = Path(self.model_dir) / "rl_agent_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config at: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def _load_tokenizer(self) -> None:
        tok_path = Path(self.model_dir) / "tokenizer" / "tokenizer.json"
        if not tok_path.is_file():
            raise FileNotFoundError(f"Missing tokenizer at: {tok_path}")
        self.tokenizer = Tokenizer.from_file(str(tok_path))

    def _init_session(self) -> None:
        model_path = Path(self.model_dir) / "model.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX model graph at: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Attempt initialization with probed providers (Dml / CoreML / CUDA -> CPU fallback)
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=self.providers,
            )
            self.active_provider = self.session.get_providers()[0]

            # Preflight verification if non-CPU provider was selected
            if self.active_provider != "CPUExecutionProvider":
                try:
                    seq_ids, marker_pos = self._build_sequence(
                        prompt="warmup",
                        question="classify",
                        qtype="choice",
                        options=[("a", "a"), ("b", "b")],
                    )
                    input_ids = np.array([seq_ids], dtype=np.int64)
                    attention_mask = np.ones((1, len(seq_ids)), dtype=np.int64)
                    marker_pos_arr = np.array([marker_pos], dtype=np.int64)
                    marker_mask = np.ones((1, len(marker_pos)), dtype=bool)
                    qtype_arr = np.array([0], dtype=np.int64)
                    self.session.run(None, {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "marker_pos": marker_pos_arr,
                        "marker_mask": marker_mask,
                        "qtype": qtype_arr,
                    })
                    logger.info(f"Verified active provider: {self.active_provider}")
                except Exception as probe_err:
                    logger.info(f"Provider {self.active_provider} failed pre-flight verification ({probe_err}); falling back to CPUExecutionProvider.")
                    self._fallback_to_cpu()
            else:
                logger.info(f"Initialized ngam ONNX session with provider: {self.active_provider}")
        except Exception as err:
            logger.warning(
                f"Failed to initialize session with requested providers {self.providers}: {err}. "
                f"Falling back to CPUExecutionProvider."
            )
            self._fallback_to_cpu()

    def _fallback_to_cpu(self) -> None:
        """Fallback session to CPUExecutionProvider."""
        model_path = Path(self.model_dir) / "model.onnx"
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.active_provider = "CPUExecutionProvider"

    def tokenize(self, text: str) -> List[int]:
        """Convert raw text to token ID sequence without special boundary tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def get_temperature(self, question_type: str, option_count: int) -> float:
        """
        Retrieve calibrated temperature scale factor from rl_agent_config.json
        based on question type and option cardinality.
        """
        if option_count <= 2:
            size_bucket = "2"
        elif option_count <= 5:
            size_bucket = "3-5"
        elif option_count <= 10:
            size_bucket = "6-10"
        else:
            size_bucket = "11+"

        key = f"{question_type}:{size_bucket}"
        by_opts = self.config.get("temperature_by_options", {})
        if key in by_opts:
            return float(by_opts[key])

        default_temps = self.config.get("temperature", [1.63, 1.25, 1.98])
        qtype_map = {"choice": 0, "score": 1, "noul": 2}
        idx = qtype_map.get(question_type, 0)
        if isinstance(default_temps, list) and idx < len(default_temps):
            return float(default_temps[idx])

        return 1.0

    def _build_sequence(
        self,
        prompt: str,
        question: str,
        qtype: Literal["choice", "score", "noul"],
        options: Sequence[Tuple[str, str]],
    ) -> Tuple[List[int], List[int]]:
        """
        Build the Laya sequence structure:
          [CLS] <qtype> question: <question> [SEP]
          [MASK] opt0 [MASK] opt1 ... [SEP]
          <prompt> [SEP]
        """
        max_len = self.config.get("max_len", 512)
        head_max_len = self.config.get("head_max_len", 192)

        # 1. Encode question header
        header_text = f"{qtype} question: {question}"
        head_ids = self.tokenize(header_text)

        # 2. Encode options with [MASK] markers
        opt_token_groups: List[List[int]] = []
        for label, desc in options:
            content = f" {label}: {desc}" if desc and desc != label else f" {label}"
            sub_ids = self.tokenize(content)[:48]
            opt_token_groups.append([self.mask_id] + sub_ids)

        # Ensure options fit within head_max_len
        opt_budget = head_max_len - sum(len(g) for g in opt_token_groups)
        if opt_budget < 16:
            per_opt = max(4, (head_max_len - 16) // max(1, len(opt_token_groups)))
            opt_token_groups = [g[:per_opt] for g in opt_token_groups]
            opt_budget = head_max_len - sum(len(g) for g in opt_token_groups)

        head_ids = head_ids[:max(8, opt_budget)]

        # Assemble head prefix
        token_ids: List[int] = [self.cls_id] + head_ids + [self.sep_id]
        marker_positions: List[int] = []
        for g in opt_token_groups:
            marker_positions.append(len(token_ids))
            token_ids.extend(g)
        token_ids.append(self.sep_id)

        # 3. Add context / state prompt
        room = max(0, max_len - len(token_ids) - 1)
        prompt_ids = self.tokenize(prompt)[:room]
        token_ids.extend(prompt_ids)
        token_ids.append(self.sep_id)

        return token_ids[:max_len], [m for m in marker_positions if m < max_len]

    def _execute_forward(
        self,
        token_ids: List[int],
        marker_positions: List[int],
        qtype_code: int,
    ) -> Tuple[np.ndarray, Optional[float], float]:
        """
        Execute ONNX inference forward pass with latency profiling and automatic provider fallback.
        """
        k = len(marker_positions)
        inputs = {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(token_ids)), dtype=np.int64),
            "marker_pos": np.array([marker_positions], dtype=np.int64),
            "marker_mask": np.ones((1, k), dtype=bool),
            "qtype": np.array([qtype_code], dtype=np.int64),
        }

        t0 = time.perf_counter()
        try:
            outputs = self.session.run(None, inputs)
        except Exception as run_err:
            if self.active_provider != "CPUExecutionProvider":
                logger.warning(
                    f"Execution failed on provider '{self.active_provider}': {run_err}. "
                    f"Falling back to CPUExecutionProvider."
                )
                self._fallback_to_cpu()
                outputs = self.session.run(None, inputs)
            else:
                raise run_err

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        logits = outputs[0]  # shape (1, >= k)
        raw_scores = logits[0, :k].astype(np.float64)

        act_probability = None
        if len(outputs) > 1 and outputs[1] is not None:
            act_logits = outputs[1][0].astype(np.float64)
            shifted_act = act_logits - np.max(act_logits)
            exp_act = np.exp(shifted_act)
            act_probs = exp_act / np.sum(exp_act)
            act_probability = float(act_probs[0])

        return raw_scores, act_probability, latency_ms

    def decide_choice(
        self,
        prompt: str,
        options: Union[List[str], Dict[str, str]],
        question: Optional[str] = None,
    ) -> Choice[str]:
        """
        Evaluate a categorical choice decision among multiple candidates.
        """
        q_text = question or "Which option is most appropriate?"
        if isinstance(options, dict):
            opt_pairs = list(options.items())
            labels = list(options.keys())
        elif isinstance(options, list):
            opt_pairs = [(str(opt), str(opt)) for opt in options]
            labels = [str(opt) for opt in options]
        else:
            raise TypeError("Options must be a list of strings or dict of {label: description}")

        if len(opt_pairs) < 2:
            raise ValueError("Choice evaluation requires at least 2 candidate options.")

        tokens, markers = self._build_sequence(prompt, q_text, "choice", opt_pairs)
        k = len(markers)
        raw_scores, act_prob, latency = self._execute_forward(tokens, markers, qtype_code=0)

        temp = self.get_temperature("choice", k)
        scaled = raw_scores / temp
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)

        winner_idx = int(np.argmax(probs))
        winner_label = labels[winner_idx]

        probs_dict = {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}
        entropy = compute_shannon_entropy(probs)
        norm_entropy = compute_normalized_entropy(probs)
        confidence = max(0.0, min(1.0, 1.0 - norm_entropy))
        ambiguous, reason = is_high_entropy(probs)
        is_ood = ambiguous
        rejection_r = reason if is_ood else None

        return Choice[str](
            type="choice",
            decision=winner_label,
            probabilities=probs_dict,
            options=labels,
            confidence=round(confidence, 4),
            entropy=round(entropy, 4),
            ambiguous=ambiguous,
            ambiguity_reason=reason,
            is_out_of_domain=is_ood,
            rejection_reason=rejection_r,
            act_probability=round(act_prob, 4) if act_prob is not None else None,
            latency_ms=round(latency, 2),
            provider=self.active_provider,
        )

    def decide_score(
        self,
        prompt: str,
        min_val: int = 0,
        max_val: int = 5,
        criteria: Optional[List[str]] = None,
        question: Optional[str] = None,
    ) -> Score:
        """
        Evaluate an ordinal or continuous score decision bounded in [min_val, max_val].
        """
        q_text = question or "Rate the level, quality, or severity of this case."

        if criteria:
            k = len(criteria)
            opt_pairs = [(f"level {i}", criteria[i]) for i in range(k)]
            levels = list(range(min_val, min_val + k))
            legend = {str(levels[i]): criteria[i] for i in range(k)}
        else:
            levels = list(range(min_val, max_val + 1))
            k = len(levels)
            opt_pairs = [(f"level {lvl}", f"Score level {lvl}") for lvl in levels]
            legend = {str(lvl): f"Level {lvl}" for lvl in levels}

        tokens, markers = self._build_sequence(prompt, q_text, "score", opt_pairs)
        raw_scores, act_prob, latency = self._execute_forward(tokens, markers, qtype_code=1)

        temp = self.get_temperature("score", k)
        scaled = raw_scores / temp
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)

        expected_score = float(np.sum(np.array(levels, dtype=np.float64) * probs))
        probs_dict = {str(levels[i]): round(float(probs[i]), 4) for i in range(k)}

        entropy = compute_shannon_entropy(probs)
        norm_entropy = compute_normalized_entropy(probs)
        confidence = max(0.0, min(1.0, 1.0 - norm_entropy))
        ambiguous, reason = is_high_entropy(probs)
        is_ood = ambiguous
        rejection_r = reason if is_ood else None

        return Score(
            type="score",
            score=round(expected_score, 4),
            probabilities=probs_dict,
            min_val=min_val,
            max_val=max_val if not criteria else (min_val + len(criteria) - 1),
            legend=legend,
            confidence=round(confidence, 4),
            entropy=round(entropy, 4),
            ambiguous=ambiguous,
            ambiguity_reason=reason,
            is_out_of_domain=is_ood,
            rejection_reason=rejection_r,
            act_probability=round(act_prob, 4) if act_prob is not None else None,
            latency_ms=round(latency, 2),
            provider=self.active_provider,
        )

    def decide_noul(
        self,
        prompt: str,
        statement: str,
    ) -> Noul:
        """
        Evaluate an epistemic verification question: does the statement hold?
        """
        q_text = statement
        opt_pairs = [
            ("false", "no, the statement does not hold"),
            ("true", "yes, the statement holds"),
        ]

        tokens, markers = self._build_sequence(prompt, q_text, "noul", opt_pairs)
        raw_scores, act_prob, latency = self._execute_forward(tokens, markers, qtype_code=2)

        temp = self.get_temperature("noul", 2)
        scaled = raw_scores / temp
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals)

        noul_val = float(probs[1])
        verdict = bool(noul_val >= 0.5)

        probs_dict = {
            "false": round(float(probs[0]), 4),
            "true": round(noul_val, 4),
        }

        entropy = compute_shannon_entropy(probs)
        norm_entropy = compute_normalized_entropy(probs)
        confidence = max(0.0, min(1.0, 1.0 - norm_entropy))
        ambiguous, reason = is_high_entropy(probs)
        is_ood = ambiguous
        rejection_r = reason if is_ood else None

        return Noul(
            type="noul",
            noul=round(noul_val, 4),
            decision=verdict,
            probabilities=probs_dict,
            confidence=round(confidence, 4),
            entropy=round(entropy, 4),
            ambiguous=ambiguous,
            ambiguity_reason=reason,
            is_out_of_domain=is_ood,
            rejection_reason=rejection_r,
            act_probability=round(act_prob, 4) if act_prob is not None else None,
            latency_ms=round(latency, 2),
            provider=self.active_provider,
        )

    def decide(
        self,
        prompt: str,
        choice: Optional[Union[List[str], Dict[str, str], str]] = None,
        score: Optional[Union[Tuple[int, int], List[str], str]] = None,
        noul: Optional[str] = None,
        question: Optional[str] = None,
    ) -> DecisionResult:
        """
        Unified polymorphic decision router.
        Auto-routes according to provided arguments.
        """
        if choice is not None:
            parsed_choice: Union[List[str], Dict[str, str]]
            if isinstance(choice, str):
                parsed_choice = [c.strip() for c in choice.split(",") if c.strip()]
            else:
                parsed_choice = choice

            res = self.decide_choice(prompt=prompt, options=parsed_choice, question=question)
            return DecisionResult(
                prompt=prompt,
                decision=res,
                latency_ms=res.latency_ms,
                provider=res.provider,
                model_name=self.config.get("model_name", "laya-onnx"),
                tokens_used=len(self.tokenize(prompt)) if self.tokenizer else 0,
                is_ambiguous=res.ambiguous,
                is_out_of_domain=res.is_out_of_domain,
                rejection_reason=res.rejection_reason,
            )

        if score is not None:
            min_v, max_v = 0, 5
            crit_list = None
            if isinstance(score, (tuple, list)) and len(score) == 2 and all(isinstance(x, int) for x in score):
                min_v, max_v = score[0], score[1]
            elif isinstance(score, list) and all(isinstance(x, str) for x in score):
                crit_list = score
            elif isinstance(score, str):
                if "-" in score:
                    parts = score.split("-")
                    min_v, max_v = int(parts[0]), int(parts[1])
                elif "," in score:
                    crit_list = [s.strip() for s in score.split(",")]
                else:
                    max_v = int(score)

            res = self.decide_score(
                prompt=prompt,
                min_val=min_v,
                max_val=max_v,
                criteria=crit_list,
                question=question,
            )
            return DecisionResult(
                prompt=prompt,
                decision=res,
                latency_ms=res.latency_ms,
                provider=res.provider,
                model_name=self.config.get("model_name", "laya-onnx"),
                tokens_used=len(self.tokenize(prompt)) if self.tokenizer else 0,
                is_ambiguous=res.ambiguous,
                is_out_of_domain=res.is_out_of_domain,
                rejection_reason=res.rejection_reason,
            )

        if noul is not None:
            res = self.decide_noul(prompt=prompt, statement=noul)
            return DecisionResult(
                prompt=prompt,
                decision=res,
                latency_ms=res.latency_ms,
                provider=res.provider,
                model_name=self.config.get("model_name", "laya-onnx"),
                tokens_used=len(self.tokenize(prompt)) if self.tokenizer else 0,
                is_ambiguous=res.ambiguous,
                is_out_of_domain=res.is_out_of_domain,
                rejection_reason=res.rejection_reason,
            )

        raise ValueError("Must provide at least one of: choice, score, or noul.")


# Decider alias for concise API usage
Decider = UniversalDecider
