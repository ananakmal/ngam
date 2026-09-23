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
    if isinstance(dec, Choice):
        lines.append("-" * 66)
        lines.append(f"Decision:   {dec.decision}")
        lines.append(f"Confidence: {dec.confidence * 100:.2f}%")
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
        lines.append(f"Score:      {dec.score:.4f} (Range: {dec.min_val} to {dec.max_val})")
        lines.append(f"Confidence: {dec.confidence * 100:.2f}%")
        lines.append(f"Entropy:    {dec.entropy:.4f} bits (Ambiguous: {dec.ambiguous})")
        lines.append("Distribution:")
        for lvl, prob in dec.probabilities.items():
            label = dec.legend.get(lvl, f"Level {lvl}") if dec.legend else f"Level {lvl}"
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            lines.append(f"  * {label:<15} {prob * 100:6.2f}%  [{bar}]")

    elif isinstance(dec, Noul):
        lines.append("-" * 66)
        verdict_str = "TRUE (Holds)" if dec.decision else "FALSE (Does not hold)"
        lines.append(f"Verdict:    {verdict_str}")
        lines.append(f"Holds Prob: {dec.noul * 100:.2f}%")
        lines.append(f"Confidence: {dec.confidence * 100:.2f}%")
        lines.append(f"Entropy:    {dec.entropy:.4f} bits (Ambiguous: {dec.ambiguous})")
        lines.append("Distribution:")
        for k, prob in dec.probabilities.items():
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            lines.append(f"  * {k:<15} {prob * 100:6.2f}%  [{bar}]")

    lines.append("=" * 66)
    return "\n".join(lines)


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not (parsed.choice or parsed.score or parsed.noul):
        parser.error("You must specify at least one decision mode: --choice, --score, or --noul")

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
