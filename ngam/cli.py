"""
ngam.cli
========
Command-line interface for the ngam Universal Neural Decision Engine.
Usage:
    ngam "The customer asks for refund" --choice "billing,tech,sales"
    ngam "Production cluster crashed" --score "0-5"
    ngam "The server is offline" --noul "The server is currently unreachable"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

# Ensure standard streams handle UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ngam.engine import UniversalDecider
from ngam.schema import Choice, Noul, Score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ngam",
        description="ngam: Universal Neural Decision Engine (Hardware-Accelerated / DirectML / CoreML / CUDA / CPU)",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        type=str,
        help="Input text, prompt, ticket, or context state to evaluate",
    )
    parser.add_argument(
        "-c", "--choice",
        type=str,
        default=None,
        help="Comma-separated candidate choices (e.g. 'billing,technical,sales')",
    )
    parser.add_argument(
        "-s", "--score",
        type=str,
        default=None,
        help="Score bounds or levels (e.g. '0-5' or 'low,medium,high,critical')",
    )
    parser.add_argument(
        "-n", "--noul",
        type=str,
        default=None,
        help="Epistemic statement to verify true/false holds (e.g. 'Customer wants refund')",
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="Explicit question to guide routing / scoring",
    )
    parser.add_argument(
        "-p", "--provider",
        type=str,
        default=None,
        help="Execution provider preference ('dml', 'coreml', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Explicitly target a specific GPU adapter index",
    )
    parser.add_argument(
        "--adapters",
        action="store_true",
        help="Inspect all DXGI display adapters, VRAM, and DirectML capabilities",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to custom model weights directory",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run 100%% offline without checking Hugging Face",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON serialized DecisionResult",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Output only the winning decision value",
    )
    return parser


def format_cli_output(result) -> str:
    lines = []
    lines.append("=" * 66)
    lines.append("  [NGAM]: Universal Neural Decision Engine")
    lines.append("=" * 66)
    lines.append(f"Prompt:     \"{result.prompt}\"")
    lines.append(f"Latency:    {result.latency_ms:.2f} ms")
    lines.append(f"Provider:   {result.provider}")
    lines.append(f"Tokens:     {result.tokens_used}")

    dec = result.decision
    is_ambig = bool(
        getattr(result, "is_ambiguous", False)
        or getattr(result, "abstain", False)
        or getattr(dec, "ambiguous", False)
        or getattr(dec, "abstain", False)
    )

    if isinstance(dec, Choice):
        lines.append("-" * 66)
        top_prob = max(dec.probabilities.values()) if dec.probabilities else 0.0
        if is_ambig:
            lines.append(f"Decision:   [ABSTAIN / AMBIGUOUS] (Top Candidate: {dec.decision})")
            lines.append(f"Certainty:  {dec.confidence * 100:.2f}% (1 - H_norm)")
            lines.append(f"Top Prob:   {top_prob * 100:.2f}% (Softmax Peak)")
            lines.append("Action:     Ambiguous decision detected. Downstream callers should abstain or escalate to human.")
        else:
            lines.append(f"Decision:   {dec.decision}")
            lines.append(f"Certainty:  {dec.confidence * 100:.2f}% (1 - H_norm)")
            lines.append(f"Top Prob:   {top_prob * 100:.2f}% (Softmax Peak)")

        lines.append(f"Entropy:    {dec.entropy:.4f} bits (Ambiguous: {dec.ambiguous})")
        if dec.ambiguity_reason:
            lines.append(f"Ambiguity:  {dec.ambiguity_reason}")
        lines.append("Distribution:")
        for opt, prob in dec.probabilities.items():
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            lines.append(f"  * {opt:<15} {prob * 100:6.2f}%  [{bar}]")

    elif isinstance(dec, Score):
        lines.append("-" * 66)
        top_prob = max(dec.probabilities.values()) if dec.probabilities else 0.0
        top_level = max(dec.probabilities.items(), key=lambda x: x[1])[0] if dec.probabilities else str(dec.score)
        candidate_label = dec.legend.get(top_level, f"Level {top_level}") if dec.legend else f"Level {top_level}"
        if is_ambig:
            lines.append(f"Decision:   [ABSTAIN / AMBIGUOUS] (Top Candidate: {candidate_label})")
            lines.append(f"Certainty:  {dec.confidence * 100:.2f}% (1 - H_norm)")
            lines.append(f"Top Prob:   {top_prob * 100:.2f}% (Softmax Peak)")
            lines.append("Action:     Ambiguous decision detected. Downstream callers should abstain or escalate to human.")
        else:
            lines.append(f"Score:      {dec.score:.4f} (Range: {dec.min_val} to {dec.max_val})")
            lines.append(f"Certainty:  {dec.confidence * 100:.2f}% (1 - H_norm)")
            lines.append(f"Top Prob:   {top_prob * 100:.2f}% (Softmax Peak)")

        lines.append(f"Entropy:    {dec.entropy:.4f} bits (Ambiguous: {dec.ambiguous})")
        if dec.ambiguity_reason:
            lines.append(f"Ambiguity:  {dec.ambiguity_reason}")
        lines.append("Distribution:")
        for lvl, prob in dec.probabilities.items():
            label = dec.legend.get(lvl, f"Level {lvl}") if dec.legend else f"Level {lvl}"
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            lines.append(f"  * {label:<15} {prob * 100:6.2f}%  [{bar}]")

    elif isinstance(dec, Noul):
        lines.append("-" * 66)
        top_prob = max(dec.probabilities.values()) if dec.probabilities else (dec.noul if dec.decision else 1.0 - dec.noul)
        candidate_str = "TRUE (Holds)" if dec.decision else "FALSE (Does not hold)"
        if is_ambig:
            lines.append(f"Decision:   [ABSTAIN / AMBIGUOUS] (Top Candidate: {candidate_str})")
            lines.append(f"Certainty:  {dec.confidence * 100:.2f}% (1 - H_norm)")
            lines.append(f"Top Prob:   {top_prob * 100:.2f}% (Softmax Peak)")
            lines.append("Action:     Ambiguous decision detected. Downstream callers should abstain or escalate to human.")
        else:
            verdict_str = "TRUE (Holds)" if dec.decision else "FALSE (Does not hold)"
            lines.append(f"Decision:   {verdict_str}")
            lines.append(f"Certainty:  {dec.confidence * 100:.2f}% (1 - H_norm)")
            lines.append(f"Top Prob:   {top_prob * 100:.2f}% (Softmax Peak)")
            lines.append(f"Holds Prob: {dec.noul * 100:.2f}%")

        lines.append(f"Entropy:    {dec.entropy:.4f} bits (Ambiguous: {dec.ambiguous})")
        if dec.ambiguity_reason:
            lines.append(f"Ambiguity:  {dec.ambiguity_reason}")
        lines.append("Distribution:")
        for k, prob in dec.probabilities.items():
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            lines.append(f"  * {k:<15} {prob * 100:6.2f}%  [{bar}]")

    lines.append("=" * 66)
    return "\n".join(lines)


def inspect_adapters(as_json: bool = False) -> None:
    """Inspect all DXGI display adapters, VRAM, and DirectML capabilities."""
    import dataclasses
    try:
        from ngam.dxgi import (
            enumerate_adapters,
            get_best_dml_device_id,
            get_hardware_adapters,
        )
    except ImportError:
        try:
            from agy_decide.dxgi import (
                enumerate_adapters,
                get_best_dml_device_id,
                get_hardware_adapters,
            )
        except ImportError:
            if as_json:
                print(json.dumps({"error": "DXGI inspection not available", "adapters": []}))
            else:
                print("DXGI hardware inspection is not available on this platform or module not found.")
            return

    all_adapters = enumerate_adapters()
    hw_adapters = get_hardware_adapters()
    best_id = get_best_dml_device_id()

    if as_json:
        payload = {
            "adapters": [dataclasses.asdict(a) for a in all_adapters],
            "hardware_adapters": [dataclasses.asdict(a) for a in hw_adapters],
            "best_dml_device_id": best_id,
        }
        print(json.dumps(payload, indent=2))
        return

    lines = []
    lines.append("=" * 72)
    lines.append("  [NGAM]: DXGI Hardware Adapters & DirectML Inspection")
    lines.append("=" * 72)
    lines.append(f"Total DXGI Adapters Detected: {len(all_adapters)}\n")
    if not all_adapters:
        lines.append("  No DXGI display adapters detected (non-Windows OS or headless VM).")
    else:
        for a in all_adapters:
            sw_str = "YES (Software)" if a.is_software else "No (Hardware)"
            disc_str = "YES (Discrete)" if a.is_discrete else "No (Integrated/WARP)"
            dml_str = "YES (Capable)" if a.is_hardware_dml_capable else "No"
            lines.append(f"  [Adapter {a.device_id}] {a.name}")
            lines.append(f"    Vendor:         {a.vendor_name} (0x{a.vendor_id:04X})")
            lines.append(f"    Subsystem ID:   0x{a.subsystem_id:04X}")
            lines.append(f"    Dedicated VRAM: {a.dedicated_vram_mb:,.1f} MB")
            lines.append(f"    Shared RAM:     {a.shared_ram_mb:,.1f} MB")
            lines.append(f"    Software:       {sw_str}")
            lines.append(f"    Discrete GPU:   {disc_str}")
            lines.append(f"    DML Capable:    {dml_str}")
            lines.append("")

    lines.append("-" * 72)
    lines.append(f"Filtered Hardware Adapters: {len(hw_adapters)}")
    for rank, a in enumerate(hw_adapters, start=1):
        tier = "dGPU" if a.is_discrete else "iGPU"
        lines.append(f"  Rank {rank}: Device {a.device_id} ({a.name}) [{tier}] — {a.dedicated_vram_mb:,.1f} MB VRAM")

    lines.append("-" * 72)
    lines.append(f"Best DirectML Device ID: {best_id}")
    if best_id is not None:
        best_adapter = next((a for a in hw_adapters if a.device_id == best_id), None)
        best_desc = f" ({best_adapter.name})" if best_adapter else ""
        lines.append(f"  -> Selected: [{best_id}]{best_desc}")
    else:
        lines.append("  -> DirectML Hardware Acceleration: Not Available (falling back to CPU)")
    lines.append("=" * 72)
    print("\n".join(lines))


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.adapters:
        inspect_adapters(as_json=parsed.json)
        return 0

    if not parsed.prompt:
        parser.error("the following arguments are required: prompt (or use --adapters)")

    if parsed.device_id is not None:
        import os
        os.environ["NGAM_DML_DEVICE_ID"] = str(parsed.device_id)
        os.environ["AGY_DML_DEVICE_ID"] = str(parsed.device_id)
        if not parsed.provider:
            parsed.provider = "dml"

    if not (parsed.choice or parsed.score or parsed.noul):
        parsed.choice = "urgent,routine,inquiry,spam"
        ansi_notice = "\033[36m[Smart Default]: Using categorical triage mode (urgent, routine, inquiry, spam)\033[0m"
        if not parsed.quiet and not parsed.json:
            print(ansi_notice)
        elif parsed.json:
            sys.stderr.write(f"{ansi_notice}\n")

    try:
        decider = UniversalDecider(
            model_dir=parsed.model_dir,
            preferred_provider=parsed.provider,
            offline_mode=parsed.offline,
        )

        result = decider.decide(
            prompt=parsed.prompt,
            choice=parsed.choice,
            score=parsed.score,
            noul=parsed.noul,
            question=parsed.question,
        )

        if parsed.json:
            print(result.model_dump_json(indent=2))
        elif parsed.quiet:
            if hasattr(result.decision, "decision"):
                print(result.decision.decision)
            elif hasattr(result.decision, "score"):
                print(result.decision.score)
            elif hasattr(result.decision, "noul"):
                print(result.decision.noul)
        else:
            print(format_cli_output(result))

        return 0
    except Exception as e:
        sys.stderr.write(f"Error in ngam: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
