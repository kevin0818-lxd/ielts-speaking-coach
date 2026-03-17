# IELTS Speaking Coach

An OpenClaw skill that acts as a professional IELTS Speaking Examiner and Tutor. Get instant, evidence-based assessment of your spoken English with actionable feedback.

## Features

- **Four-criterion scoring** (Band 1-9, 0.5 increments): Fluency & Coherence, Lexical Resource, Grammar, Pronunciation — each with specific transcript evidence
- **CHAI-calibrated scores**: Human-prior calibration adjusts for Part difficulty and criterion bias (Polasa et al., 2025)
- **Grammar corrections**: Identifies genuine spoken grammar errors with minimal corrections and rule explanations
- **Context-aware vocabulary upgrades**: Topic-specific collocational improvements, not generic thesaurus swaps
- **Spoken-register model answers**: Rewrites at your target band using natural discourse markers, contractions, and idiomatic language
- **All three Parts**: Part 1 (Interview), Part 2 (Long Turn), Part 3 (Discussion) with Part-specific expectations

## Installation

```bash
openclaw skills install ielts-speaking-coach
```

## Quick Start

### Assess a response

> Score my IELTS Part 1 answer: "I'm studying computer science at university. It's quite challenging but I find it rewarding because I enjoy problem-solving."

The skill returns a full assessment: scores, grammar corrections, vocabulary upgrades, model answer, and study tips.

### Practice session

> Let's do IELTS speaking practice, Part 2

The skill gives you an IELTS-style cue card, listens to your response, and provides detailed feedback.

### Quick score only

> Quick score: "Technology has changed our lives significantly. For instance, we can now communicate with anyone around the world instantly."

Returns only the score breakdown table — no corrections or model answer.

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `target_band` | number | 7.0 | Your target IELTS band (model answers aim at this level) |
| `default_part` | number | 1 | Default IELTS Speaking part (1, 2, or 3) |
| `deepseek_api_key` | string | — | Optional API key for enhanced grammar analysis |
| `local_model_path` | string | `Qwen/Qwen3-0.6B` | Base model path for local model answer generation |
| `adapter_path` | string | — | Path to fine-tuned LoRA adapter for spoken-register model answers |

```bash
openclaw skills config ielts-speaking-coach target_band 7.5
openclaw skills config ielts-speaking-coach default_part 2
```

## Local Model Answer Generation (Apple Silicon)

This skill includes a fine-tuned Qwen3-0.6B model specifically trained for IELTS spoken-register model answer generation. It produces more natural spoken English than general-purpose LLMs because it was trained in two stages:

1. **Stage 1 — Domain Adaptation**: LoRA warm-up on IELTS candidate responses and academic discussion transcripts
2. **Stage 2 — Model Answer SFT**: Supervised fine-tuning on teacher-generated spoken-register model answers

Inference uses max 768 tokens per response (`max_seq_length`); this is the generation limit, not the training sample count. See adapter training logs for the actual number of training examples.

### Setup

Requirements: Apple Silicon Mac, Python 3.9+, mlx-lm

```bash
pip install mlx-lm
```

Configure the model paths:

```bash
openclaw skills config ielts-speaking-coach local_model_path "Qwen/Qwen3-0.6B"
openclaw skills config ielts-speaking-coach adapter_path "/path/to/your/adapter/directory"
```

The adapter directory should contain `adapters.safetensors` and `adapter_config.json` from the training output.

### Standalone Usage

You can also run the model answer generator directly:

```bash
python scripts/generate_model_answer.py \
  --transcript "I am a student. I study business. It is good." \
  --part 1 \
  --user-band 6.0 \
  --adapter-path /path/to/adapter \
  --json
```

When configured, the skill automatically uses the local model for model answer generation and falls back to the built-in LLM if unavailable.

## How Scoring Works

The skill evaluates four criteria independently using official IELTS Band Descriptors:

1. **Fluency & Coherence**: Speech rate (WPM), filler density, discourse markers, coherence
2. **Lexical Resource**: Vocabulary sophistication, semantic variety, idiomatic usage, collocations
3. **Grammatical Range & Accuracy**: Complex structures per 18 words, error density
4. **Pronunciation**: Self-rated or estimated at 6.0 in text-only mode

Scores are calibrated using the CHAI framework (hybrid human-prior calibration) with Part-specific adjustments, then averaged and rounded to the nearest 0.5 band.

## Scoring Accuracy

This skill uses prompt-driven assessment — it is a study tool, not a replacement for a human examiner. Scores are most accurate for:

- Band 5.0 to 8.0 responses
- Part 1 and Part 3 (text-based criteria)
- Grammar and Lexical Resource (directly observable in text)

Pronunciation scoring requires audio input and defaults to 6.0 in text-only mode.

## License

MIT
