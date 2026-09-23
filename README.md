# `ngam` ⚡
### Ultra-Fast Typed Neural Decision Engine
*Calibrated Probabilistic Routing, Scoring, and Epistemic Verification with Shannon Entropy Gating across Windows, macOS, and Linux.*

[![CI](https://github.com/ananakmal/ngam/actions/workflows/ci.yml/badge.svg)](https://github.com/ananakmal/ngam/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17+-green.svg)](https://onnxruntime.ai/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-orange.svg)](https://docs.pydantic.dev/)

---

## 🚀 Overview

**`ngam`** (*Malay for "exact fit", "just right", "on point"*) is a standalone, hardware-accelerated neural decision engine designed specifically for modern autonomous agents, multi-model orchestrators, and high-throughput production backends.

Rather than burning 1,000+ tokens and waiting 800ms to 3,000ms for a cloud Large Language Model (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0) to perform simple routing, scoring, or truth verification, **`ngam` intercepts incoming queries locally with zero cloud API token costs**:
- **In-memory CPU daemon inference (~340–450ms)**: ModernBERT-large is 421M parameters, which takes ~350ms of pure tensor math on multi-threaded AVX2/AVX-512 CPUs.
- **Hardware-accelerated GPU inference (~15–35ms)**: Massive tensor parallelism on Linux CUDA / TensorRT backends.
- **HTTP daemon keep-alive connection (<5ms overhead)**: Persistent localhost daemon eliminating process startup overhead.

It combines a calibrated INT8 ONNX transformer backbone with **Shannon Entropy ambiguity gating**, ensuring that routine deterministic decisions are executed locally with high confidence, while ambiguous or out-of-distribution queries are reliably escalated to frontier reasoning models.

---

## 🧠 System-1 vs. System-2 Decision Routing

Cognitive science divides reasoning into two complementary regimes (Kahneman, 2011):
- **System-1 (Fast, Reflexive, Parallel)**: Immediate perceptual classification, routing, and gut checks.
- **System-2 (Slow, Deliberate, Sequential)**: Multi-step reasoning, architectural planning, and deep code generation.

Today's AI agent stacks suffer from an architectural defect: **they use System-2 LLMs for System-1 tasks**. Asking a 70B+ parameter model *"Should this ticket go to billing or tech support?"* is slow, costly, non-deterministic, and prone to hallucinations.

`ngam` serves as the **System-1 cortex** for your AI infrastructure:

```
                            [ Incoming User Request / Event ]
                                           │
                                           ▼
                            ┌───────────────────────────────┐
                            │      ngam Decision Engine     │
                            │ (Local ONNX, ~15-35ms GPU /   │
                            │       ~340-450ms CPU)         │
                            └──────────────┬────────────────┘
                                           │
                   ┌───────────────────────┴───────────────────────┐
                   │                                               │
      Low Entropy: H(P) < Cutoff                      High Entropy: H(P) >= Cutoff
         (Decisive, High Confidence)                   (Ambiguous / Out-of-Distribution)
                   │                                               │
                   ▼                                               ▼
     ┌───────────────────────────┐                   ┌───────────────────────────┐
     │   Local Reflex Execution  │                   │ Escalation to System-2    │
     │   * Direct Action / Route │                   │ * Cloud LLM / Deep Reason │
     │   * Zero token cost       │                   │ * Fallback Council        │
     │   * Fast local response   │                   │ * Human-in-the-loop       │
     └───────────────────────────┘                   └───────────────────────────┘
```

---

## 🏛️ Architecture & Decision Interception

`ngam` evaluates input states using a single, unified bidirectional transformer pass with specialized sequence markers and calibrated temperature scaling.

```
       Input Prompt: "I was billed twice on my invoice. Please refund the duplicate charge."
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Sequence Packing:                                                                      │
│ [CLS] choice question: Which option is most appropriate? [SEP]                        │
│ [MASK] billing: billing [MASK] technical: technical [MASK] sales: sales [SEP]         │
│ I was billed twice on my invoice. Please refund the duplicate charge. [SEP]            │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   Hardware Dispatcher   │
                       │   (DirectML/CoreML/CUDA)│
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  Logits & Head Scoring  │
                       │  Temperature Calibrated │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐        ┌──────────────────┐       ┌─────────────────┐
│ Calibrated Softmax│       │  Shannon Entropy │       │ Epistemic Gate  │
│ billing: 0.9406 │        │  H(P) = 0.3715 b │       │ is_ood: False   │
│ tech:    0.0458 │        │  H_norm: 0.2344  │       │ Rejection: None │
│ sales:   0.0137 │        │  Confidence: 76% │       │                 │
└─────────────────┘        └──────────────────┘       └─────────────────┘
```

---

## 📦 The 3 Core Primitives

`ngam` standardizes all agent decisions into three strongly-typed Pydantic V2 contracts:

### 1. `Choice[T]` — Categorical Routing
Categorical classification across an arbitrary set of candidate labels or `{label: description}` pairs.
- Returns the winning label with calibrated probabilities summing to 1.0.
- Calculates Shannon entropy and confidence.

```python
from ngam import Decider

decider = Decider()
result = decider.decide_choice(
    prompt="Production Kubernetes cluster in us-east-1 is throwing 502 Bad Gateway errors.",
    options=["devops", "billing", "frontend", "security"]
)
print(result.decision)       # "devops"
print(result.confidence)     # 0.8841 (88.4%)
print(result.probabilities)  # {'devops': 0.942, 'frontend': 0.031, 'security': 0.021, 'billing': 0.006}
```

### 2. `Score[Min, Max]` — Continuous & Ordinal Evaluation
Evaluates quality, priority, or severity over a continuous or bounded ordinal range $[Min, Max]$.
- Calculates the mathematical expectation $\mathbb{E}[S] = \sum_{i} i \cdot p_i$.
- Supports semantic criteria definitions for rubric grading.

```python
score_res = decider.decide_score(
    prompt="Complete database storage volume corruption with zero backups available.",
    min_val=0,
    max_val=5,
    question="Rate the incident severity from 0 (trivial) to 5 (existential disaster)."
)
print(score_res.score)       # 4.892 (near maximum severity)
print(score_res.confidence)  # 0.912
```

### 3. `Noul` — Binary Epistemic Verification
Evaluates whether a given proposition or epistemic truth statement holds given the context state.
- Output: `noul` $\in [0.0, 1.0]$ representing $P(\text{holds})$.
- Boolean verdict: `True` if $P \ge 0.5$, else `False`.

```python
noul_res = decider.decide_noul(
    prompt="User account was deleted on 2026-08-12 per GDPR erasure request.",
    statement="The user's personal data is still active in production systems."
)
print(noul_res.decision)  # False
print(noul_res.noul)      # 0.041 (4.1% probability of holding)
```

---

## 📐 Shannon Entropy Ambiguity Gating & OOD Rejection

A critical flaw of standard classification models is overconfidence on garbage or out-of-domain inputs. `ngam` mathematically guarantees safety by computing the **Shannon Entropy** over the calibrated posterior distribution:

$$H(P) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

Normalized entropy maps the measure to the range $[0.0, 1.0]$:

$$H_{\text{norm}}(P) = \frac{H(P)}{\log_2(k)}$$

### Epistemic Rejection Thresholds:
1. **Near-Uniform Entropy Cutoff ($H_{\text{norm}} \ge 0.85$)**: When the model assigns roughly equal probabilities across candidates, it signals epistemic uncertainty. `ngam` flags `ambiguous = True` and `is_out_of_domain = True`.
2. **Narrow Candidate Margin ($p_{\text{top1}} - p_{\text{top2}} < 0.05$ with low confidence)**: When top candidates cannot be reliably separated ($p_{\text{top1}} < 1.5 / k$), `ngam` flags the decision as ambiguous for human or System-2 review.
3. **Decisive Winner Confidence Floor ($p_{\text{top1}} < \max(0.30, 1.25 / k)$)**: Decisions where the winning candidate fails to secure decisive probability mass (e.g. < 30%) are automatically flagged as ambiguous / out-of-domain (`ambiguous = True`), preventing low-confidence guesses from passing as definitive decisions.

```python
# Unintelligible or adversarial out-of-domain input
res = decider.decide(
    prompt="xj99-alpha qwertyuiop 77812 lorem ipsum garbage",
    choice=["billing", "technical", "sales"]
)
print(res.is_ambiguous)       # True
print(res.is_out_of_domain)   # True
print(res.rejection_reason)   # "High Shannon entropy (H_norm=0.982 >= 0.85); distribution is near-uniform."
```

---

## ⚡ Hardware Acceleration Matrix

`ngam` features an automatic platform resolver that inspects available execution providers and automatically selects the highest-performance acceleration backend:

| Operating System | Primary Acceleration Backend | Fallback | Supported Hardware | Status |
|:---|:---|:---|:---|:---|
| **Windows 11 / 10** | **CPU (`CPUExecutionProvider`)** | Single-threaded CPU | Multi-threaded AVX2 / AVX-512 CPUs | ✅ Primary Out-of-the-Box Verified (~340–450ms) |
| **Linux (Ubuntu / Debian / RHEL)** | **CUDA / TensorRT (`CUDAExecutionProvider`)** | CPU (OpenMP / oneDNN) | NVIDIA GPUs (RTX, A100, H100) | ✅ Full GPU Acceleration (~15–35ms) |
| **macOS (Apple Silicon)** | **Core ML (`CoreMLExecutionProvider`)** | CPU (Apple Accelerate) | M1 / M2 / M3 / M4 Neural Engine & Metal GPU | ✅ Hardware Accelerated (~25–50ms) |
| **Windows 11 (DirectML)** | **DirectML (`DmlExecutionProvider`)** | Auto-fallback to CPU | DirectX 12 GPUs | ⚠️ Experimental (Upstream Reshape op limitation) |

> [!NOTE] Windows DirectML Compatibility Notice
> DirectML on Windows DirectX 12 is currently experimental due to an upstream ONNX Runtime Reshape operator kernel limitation (`node_view` status code failure) when processing ModernBERT dynamic attention heads. `ngam`'s automated preflight probe catches this driver/kernel incompatibility gracefully and safely falls back to multi-threaded `CPUExecutionProvider` without crashing your application. On Windows out-of-the-box, `CPUExecutionProvider` with AVX2/AVX-512 is the primary verified execution provider. On Linux, `CUDAExecutionProvider` provides verified full GPU acceleration (~15–35ms).

### Preflight Verification Probe
During initialization, if a non-CPU execution provider is selected, `ngam` executes a calibrated 2-option preflight probe. If the underlying GPU or DirectML driver encounters an incompatibility, `ngam` catches the exception and falls back to `CPUExecutionProvider` without crashing your application.

---

## 📊 Empirical Performance & Benchmarks

The following benchmarks are derived directly from local measurements using ModernBERT-large (421M parameters, INT8 quantized graph ~210MB):

| Benchmark Phase / Operation | Execution Provider | Measured Latency | Throughput / Overhead | Real-World Operational Context |
|:---|:---|:---|:---|:---|
| **Cold Init (Session & Tokenizer Load)** | `CPUExecutionProvider` | ~1,300–1,500 ms | One-time startup | Graph deserialization & ONNX session setup from local cache |
| **Choice / Route Inference (Warm)** | CPU (multi-threaded AVX2 / AVX-512) | **~340–450 ms** (mean ~368 ms) | ~2.5–3.0 req/s | Pure tensor math for 421M ModernBERT parameters |
| **Noul Epistemic Inference (Warm)** | CPU (multi-threaded AVX2 / AVX-512) | **~520–735 ms** | ~1.5–2.0 req/s | Sequence pair verification across full context |
| **Choice / Route Inference (GPU)** | Linux CUDA (`CUDAExecutionProvider`) | **~15–35 ms** | ~30–65 req/s | Full tensor core hardware parallelization |
| **Daemon Keep-Alive Connection Overhead** | HTTP Keep-Alive / Loopback | **< 3–5 ms** | > 1,000 conn/s | Localhost HTTP socket & serialization overhead (excluding inference) |
| **Daemon Working Set Memory** | Resident RAM | ~650–850 MB | Static resident RAM | INT8 graph weights + ONNX runtime workspace buffers |

---

## 🛠️ Installation

```bash
# Core package (CPU / Core ML on macOS)
pip install ngam

# Windows DirectML GPU / NPU acceleration
pip install ngam[directml]

# Linux NVIDIA CUDA acceleration
pip install ngam[cuda]
```

---

## 💻 Quickstart Guide

### 1. Python API

```python
from ngam import Decider, Choice, Score, Noul

# 1. Initialize decider (auto-probes hardware, caches weights locally)
decider = Decider()

# 2. Unified Polymorphic Decide
# Categorical choice routing:
res_choice = decider.decide(
    prompt="The client wants to upgrade to an enterprise annual contract.",
    choice=["support", "sales", "billing"]
)
print(f"Route: {res_choice.decision.decision} ({res_choice.latency_ms:.2f}ms)")

# Ordinal score evaluation:
res_score = decider.decide(
    prompt="Production database latency spiked to 4,500ms and requests are timing out.",
    score="0-5"
)
print(f"Severity: {res_score.decision.score} / 5.0")

# Epistemic truth verification:
res_noul = decider.decide(
    prompt="Server returned HTTP 200 OK with payload {'status': 'healthy'}",
    noul="The server is responding normally."
)
print(f"Statement holds: {res_noul.decision.decision} (P={res_noul.decision.noul:.4f})")
```

### 2. Command Line Interface (CLI)

The `ngam` CLI binary provides an intuitive interface for shell scripting and operational workflows:

```bash
# Categorical Choice
ngam "Customer is disputing an unauthorized charge on credit card" --choice "billing,tech,sales"

# Ordinal Rating
ngam "System memory consumption reached 99.2%" --score "0-5"

# Epistemic Verification
ngam "Disk usage is 94%" --noul "Disk space is running dangerously low"

# Clean JSON output for jq pipeline integration
ngam "Customer inquiry" --choice "sales,support" --json | jq .decision.decision

# Fast quiet output (prints only winning value)
ngam "Reset my password" --choice "security,billing" --quiet
# Output: security

# Force 100% offline mode
ngam "Verify state" --choice "ok,error" --offline
```

---

## 🌐 In-Memory HTTP Daemon (`ngam-daemon`)

For maximum performance, run `ngam` as an in-memory HTTP daemon on port `8045`. This pre-warms model weights in RAM/VRAM, eliminating process startup overhead and delivering **sub-5ms connection overhead** (<5ms HTTP keep-alive) directly to the in-memory engine.

### Launching the Daemon

```bash
ngam-daemon --host 127.0.0.1 --port 8045
```

### Endpoints Specification

#### 1. `GET /healthz`
Returns daemon health status, active execution provider, and pre-warm verification.

```bash
curl -s http://127.0.0.1:8045/healthz
```
```json
{
  "status": "ok",
  "provider": "CPUExecutionProvider",
  "model": "ngam-onnx",
  "warm": true
}
```

#### 2. `POST /v1/decide`
Full decision routing endpoint with support for CORS and keep-alive connections.

```bash
curl -X POST http://127.0.0.1:8045/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I was billed twice on my invoice. Please refund the duplicate transaction.",
    "choice": ["billing", "technical", "sales"]
  }'
```

**Response:**
```json
{
  "prompt": "I was billed twice on my invoice. Please refund the duplicate transaction.",
  "decision": {
    "type": "choice",
    "decision": "billing",
    "probabilities": {
      "billing": 0.9406,
      "technical": 0.0458,
      "sales": 0.0137
    },
    "options": ["billing", "technical", "sales"],
    "confidence": 0.7656,
    "entropy": 0.3715,
    "ambiguous": false,
    "ambiguity_reason": null,
    "is_out_of_domain": false,
    "rejection_reason": null,
    "act_probability": 0.125,
    "latency_ms": 368.55,
    "provider": "CPUExecutionProvider"
  },
  "latency_ms": 368.55,
  "provider": "CPUExecutionProvider",
  "model_name": "ngam-onnx",
  "tokens_used": 15,
  "is_ambiguous": false,
  "is_out_of_domain": false,
  "rejection_reason": null
}
```

---

## 🐳 Docker 1-Command Deployment

Run the containerized `ngam` daemon anywhere with zero external dependencies:

```bash
# Build the production image
docker build -t ngam:latest .

# Run the daemon on port 8045
docker run -d --name ngam-service -p 8045:8045 --restart unless-stopped ngam:latest

# Verify health
curl http://localhost:8045/healthz
```

---

## 🔬 Model Transparency & Specifications

- **Backbone Architecture**: ModernBERT-large (421M parameters) with custom bidirectional sequence classification heads.
- **Quantization**: INT8 calibrated ONNX runtime graph (`~210MB` disk footprint).
- **Context Length**: Up to 512 tokens (192 tokens reserved for question and candidate option descriptions).
- **Storage Location**: `~/.cache/ngam/models/ngam_onnx/`
- **100% Offline & Private**: Zero telemetry, zero external network calls during inference. Data never leaves your machine.

---

## 🧪 Comprehensive Test Suite

`ngam` includes a 22-point test suite covering cross-platform provider probing, Shannon entropy math, Pydantic contracts, offline cache validation, deterministic mock inference, CORS preflight headers, and end-to-end inference:

```bash
python -m pytest tests/test_ngam.py -v
```

---

## 📄 License

Released under the **MIT License**. Copyright (c) 2026 **Anan Akmal**.
