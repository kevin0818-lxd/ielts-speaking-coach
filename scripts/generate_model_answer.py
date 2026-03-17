#!/usr/bin/env python3
"""
Local model answer generator using fine-tuned Qwen3-0.6B via MLX.

Features: 30s inference timeout (signal.alarm on Unix, threading on Windows),
specific error handling for model load failures (disk full, corrupted files,
permission denied). Use --timeout N to override.

Usage:
    python scripts/generate_model_answer.py \
        --transcript "I am a student. I study business." \
        --part 1 \
        --user-band 6.0 \
        --model-path Qwen/Qwen3-0.6B \
        --adapter-path /path/to/adapters.safetensors

Environment variables (alternative to CLI args):
    IELTS_MODEL_PATH    - HuggingFace model ID or local path
    IELTS_ADAPTER_PATH  - Path to LoRA adapter directory
"""

import argparse
import json
import os
import re
import signal
import sys
import threading


def build_messages(transcript: str, part: int, user_band: float) -> list:
    target_band = min(9.0, user_band + 1.0)
    target_desc = f"Band {target_band:.1f}"
    part_label = f"Part {part}"

    system_prompt = f"""You are an expert IELTS Speaking Examiner and Tutor.
The candidate currently scores around Band {user_band:.1f}. Your task is to rewrite their response into a {target_desc} Model Answer that sounds like natural, fluent SPOKEN English — not written prose.
Aim precisely at {target_desc} quality — do not overshoot to Band 9 if the target is lower.

GUIDELINES:
1. Preserve the original core meaning and opinion. Do not invent completely new facts unless the original was too vague.
2. Use SPOKEN register vocabulary: favour idiomatic collocations and phrasal verbs (e.g. "come across", "figure out") over formal/academic words. Avoid overly literary or essay-style diction.
3. Use spoken grammar norms: contractions (I'm, don't, it's, I'd), shorter clause chains, occasional fronting ("The food there, it was incredible"), tag questions, and ellipsis.
4. Include natural discourse markers: openers (Well, Actually, To be honest), connectors (so, I mean, you know, basically), hedges (I'd say, I suppose, sort of, kind of).
5. Add spoken discourse bundles: "the thing is", "what I mean is", "come to think of it", "first of all".
6. Length by part: Part 1 = 2-4 sentences, Part 2 = ~2 mins speech, Part 3 = in-depth discussion.
7. Output ONLY the rewritten spoken text. No explanations or labels. Do not use <think> or chain-of-thought. Respond directly with the model answer."""

    user_content = f"""Candidate's Response ({part_label}):
"{transcript}"

Rewrite this as a {target_desc} Model Answer."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def extract_answer(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    for prefix in [
        "Here is the rewritten",
        "Here's the rewritten",
        "Model Answer:",
        "Rewritten:",
    ]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix) :].strip().lstrip(":\n")
    raw = raw.strip('"').strip()
    raw = re.sub(r"<\|.*?\|>", "", raw).strip()
    return raw


class InferenceTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise InferenceTimeoutError("Model inference timed out")


def _run_with_timeout(func, timeout_sec: int):
    """Run func(); raise InferenceTimeoutError if it exceeds timeout_sec."""
    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = func()
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise InferenceTimeoutError(f"Model inference timed out ({timeout_sec}s)")
    if exc[0] is not None:
        raise exc[0]
    return result[0]


def main():
    parser = argparse.ArgumentParser(description="Generate IELTS model answer with local MLX model")
    parser.add_argument("--transcript", required=True, help="Candidate's spoken response text")
    parser.add_argument("--part", type=int, default=1, choices=[1, 2, 3], help="IELTS Speaking part")
    parser.add_argument("--user-band", type=float, default=6.0, help="Candidate's current band estimate")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("IELTS_MODEL_PATH", "Qwen/Qwen3-0.6B"),
        help="HuggingFace model ID or local path to base model",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("IELTS_ADAPTER_PATH", ""),
        help="Path to LoRA adapter directory (optional)",
    )
    parser.add_argument("--max-tokens", type=int, default=768, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--timeout", type=int, default=30, help="Inference timeout in seconds (default 30)")
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        print("Error: mlx-lm is not installed. Install with: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)

    adapter = args.adapter_path if args.adapter_path else None
    try:
        model, tokenizer = load(args.model_path, adapter_path=adapter)
    except FileNotFoundError as e:
        print(f"Error: Model or adapter file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        err_msg = str(e).lower()
        if "no space" in err_msg or "disk" in err_msg or "enospc" in err_msg:
            print(f"Error: Disk full or insufficient space when loading model: {e}", file=sys.stderr)
        elif "permission" in err_msg or "eacces" in err_msg:
            print(f"Error: Permission denied accessing model files: {e}", file=sys.stderr)
        else:
            print(f"Error: Failed to load model (I/O error): {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "corrupt" in err_msg or "invalid" in err_msg or "format" in err_msg:
            print(f"Error: Model or adapter file may be corrupted: {e}", file=sys.stderr)
        else:
            print(f"Error: Failed to load model (runtime error): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    messages = build_messages(args.transcript, args.part, args.user_band)

    try:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        prompt_text = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        prompt_text += "\n\nASSISTANT:\n"

    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    def _do_generate():
        return generate(
            model,
            tokenizer,
            prompt=prompt_text,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )

    try:
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(args.timeout)
        try:
            if hasattr(signal, "SIGALRM"):
                raw_output = _do_generate()
            else:
                raw_output = _run_with_timeout(_do_generate, args.timeout)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
    except InferenceTimeoutError:
        print(f"Error: Model inference timed out ({args.timeout}s)", file=sys.stderr)
        sys.exit(1)
    except (OSError, RuntimeError) as e:
        print(f"Error: Inference failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    answer = extract_answer(raw_output)

    if args.json:
        result = {
            "model_answer": answer,
            "target_band": min(9.0, args.user_band + 1.0),
            "part": args.part,
            "model": args.model_path,
            "adapter": adapter or "none",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(answer)


if __name__ == "__main__":
    main()
