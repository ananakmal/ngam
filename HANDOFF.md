# HANDOFF — ngam local validation

## 23/09/2026 — First local validation pass (clone + smoke test)

### What was done
- Repo was already cloned at `C:\dev\ngam` (clean working tree, `origin/main` @ `7e584e9`, nothing to push/pull).
- Created a fresh venv: `python -m venv .venv` using system Python **3.14.3** (`C:\Python314\python.exe`).
- `pip install -e .` — **succeeded cleanly**, no errors. Resolved: `onnxruntime 1.30.0`, `tokenizers 0.23.2`, `pydantic 2.13.5`, `numpy 2.5.3`, `huggingface_hub 1.32.0` (all wheels, no source builds needed on 3.14).
- Ran all 4 README examples + the OOD/ambiguity example + 3 domain-specific examples (Malaysia healthcare-IT / dev-flow style prompts) via a real `Decider()` instance — no mocking.
- Confirmed the CLI entry points (`ngam`, `ngam-daemon` from `[project.scripts]`) actually resolve and run, not just the Python import path.
- Ran a side diagnostic: installed the optional `ngam[directml]` extra to test the README's DirectML acceleration claim, then reverted the venv back to the plain `pip install -e .` state (matching the task's install instructions) once the diagnostic was done.
- No source files under `ngam/` were modified. No PR opened, nothing pushed.

### Pass/fail state

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `pip install -e .` | **PASS** | Clean install, Python 3.14.3, all deps incl. `huggingface_hub` resolved automatically |
| 2 | README Choice example | **PASS** (ran without error) | `decision=frontend`, `confidence=0.2064` — see calibration concern below |
| 3 | README Score example | **PASS** | `score=4.6308`, `confidence=0.7321` |
| 4 | README Noul example | **PASS** | `decision=False`, `noul=0.0` |
| 5 | README OOD/ambiguity gate | **PASS** | Correctly flagged garbage input: `is_ambiguous=True, is_out_of_domain=True`, reason "High Shannon entropy (H_norm=0.973 >= 0.85)" |
| 6 | Domain Choice 1 (MRN duplicate) | **PASS** (ran without error) | Picked plausible label but only 12.3% confidence, not flagged ambiguous |
| 7 | Domain Choice 2 (prettier/pre-commit) | **PASS** (ran without error) | Picked plausible label but only 9.7% confidence, not flagged ambiguous |
| 8 | Domain Score (council REJECT severity) | **PASS** (ran without error) | Score came back `0.9941` on a 0–5 scale — see calibration concern below |
| 9 | CLI entry point (`ngam --help`, `python -m ngam --help`, `ngam-daemon --help`, and a real functional `ngam -c ...` call) | **PASS** | Exit code 0 on all, output well-formed |

All 9 checks executed without exceptions/crashes. "PASS" above means the code path ran and returned a well-formed, schema-valid result — it does **not** mean the decision was necessarily the "right" one; see discrepancies below.

### Real measured latency

Model files were already present in `~/.cache/ngam/models/ngam_onnx` (579 MB `model.onnx.data`, timestamped earlier the same day, from repo `inferenceprince/laya-onnx-int8` on Hugging Face) — so this session's runs did **not** trigger a fresh network download; the cache-population path was already exercised previously. The complete, correctly-sized files on disk confirm the HF download mechanism does work, just not freshly exercised in this session.

- `Decider()` init (model+tokenizer load, ONNX session creation) with warm cache: **~1.3–1.5 s**
- Per-call latency (5x repeated, `CPUExecutionProvider`, README Choice prompt): wall-clock **341–455 ms**, and the engine's own internal `latency_ms` field (pure `onnxruntime.InferenceSession.run()` time) is effectively identical (**340–454 ms**) — Python/tokenization overhead is negligible; inference itself is the cost.
- Noul call: **520–735 ms** per call.
- Score/Choice calls across the full smoke test ranged **340–925 ms**.

**This is roughly 70–190x higher than the README's claimed "sub-5ms" latency**, measured purely on CPU (see DirectML finding below for why CPU was the active provider).

### DirectML claim — does not work as advertised on this machine

- The default `pip install -e .` pulls plain `onnxruntime` (CPU-only + AzureExecutionProvider), **not** `onnxruntime-directml`. `ort.get_available_providers()` → `['AzureExecutionProvider', 'CPUExecutionProvider']`. DirectML is gated behind the optional `ngam[directml]` extra, which is not installed by default despite the README/CLI banner marketing "DirectML" as if it's the out-of-box Windows behavior.
- Diagnostic: installed `pip install -e ".[directml]"` → pulled `onnxruntime-directml 1.24.4`, exposing `DmlExecutionProvider`. However, `ngam/engine.py`'s own pre-flight verification (`_init_session`) then **failed** when actually exercising the DML provider:
  ```
  onnxruntime::ExecuteKernel] Non-zero status code returned while running Reshape node.
  Name:'node_view' ... DmlExecutionProvider ... Exception(1) ... 8007005 The parameter is incorrect.
  ```
  The engine's own fallback logic caught this and silently downgraded to `CPUExecutionProvider` (this defensive design works correctly — no crash), but it means DirectML acceleration is currently **non-functional** on this machine/onnxruntime-directml 1.24.4 combo — likely an ONNX opset/op export incompatibility with the DML EP, not a Python packaging problem. This is a source/model-export issue, out of scope to fix here (read-only validation, no source changes made).
- Venv was reverted to the plain `pip install -e .` state afterward (uninstalled `onnxruntime-directml`, reinstalled plain `onnxruntime==1.30.0`) so the environment matches what the task asked for.

### Confidence/calibration observations (domain examples)

- The Kubernetes/502 README example picked `frontend` (56.8%) over `devops` (23.1%) for a textbook infra/devops prompt, with overall `confidence=20.64%` — low confidence, and the ambiguity gate did **not** trigger (margin and normalized-entropy thresholds weren't crossed), even though a human would likely flag this as a wrong or at least highly uncertain routing decision.
- Both domain Choice examples (MRN duplicate, prettier/pre-commit) picked plausible-looking winners but with single-digit-to-low-double-digit confidence (9.7%, 12.3%) and were not flagged ambiguous — the ambiguity gate's thresholds (`H_norm >= 0.85` or `margin < 0.05`) appear to leave a wide band of genuinely low-confidence decisions unflagged.
- The domain Score example (council REJECT severity, 0–5 scale) returned `0.9941` — i.e., "essentially ignore" — for a scenario described as "one REJECT after two rounds of fixes," which a reasonable operator might expect to land closer to mid-scale (2–4), not near 0. Confidence was moderate (42%).
- These are single-sample observations on out-of-training-distribution domain text (Boss's ops/governance vocabulary, not the model's likely training domain) — not a statistically rigorous calibration audit. Read as directional signal only.

### Next steps
- This was a **first local validation pass only** — proof that the package installs, runs, and the CLI/library surface work end-to-end on this machine. It is **not** a production integration, not a calibration study, and not a verdict on whether ngam is fit for Boss's actual routing use cases.
- If ngam is to be used for real decision routing (e.g., dev-flow triage, council severity scoring), it needs: (a) a proper calibration benchmark with labeled domain examples before trusting confidence/ambiguity outputs, (b) investigation of the DirectML Reshape-node failure if GPU acceleration is actually wanted on Windows, (c) a decision on whether ~350–900ms CPU latency is acceptable given the README's sub-5ms claim is not achieved on this hardware/provider combo.
- No further action taken this session; repo untouched otherwise.

---
