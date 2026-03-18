# Changelog

## 1.0.1 (2026-03)

### Updated
- Added fixed-entry IELTS speaking mode with explicit enter/exit phrases
- Added menu-driven workflows for Part 1, Part 2, Part 3, scoring, and grammar correction
- Standardized scoring and correction outputs with fixed Chinese templates
- Prioritized Part 2 cue cards from `cue-cards-2025-may-aug.md`, then `cue-cards.md`
- Added fixed topic rotation for Part 1 and Part 3 follow-up generation based on Part 2 themes

## 1.0.0 (2025-03)

### Added
- Four-criterion IELTS scoring (FC, LR, GRA, PR) with Band 1-9 descriptors
- CHAI calibration for Part-specific score adjustment
- Grammar corrections (genuine errors only, no punctuation/style)
- Context-aware vocabulary upgrades via vocab-map.json
- Spoken-register model answers (LLM or local Qwen3-0.6B)
- Part 1/2/3 support with Part-specific expectations
- 15 cue cards from 2025 May–August IELTS question bank
- Optional persistence (load_trajectory, update_trajectory) for learning path tracking
- 24 trigger keywords (English + Chinese)
