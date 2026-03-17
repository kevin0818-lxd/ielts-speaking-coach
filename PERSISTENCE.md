# Persistence (Learning Trajectory)

When `persistence_enabled` is true in skill config, the skill reads and writes learner state to the shared SQLite database at `backend/data/learner_trajectory.db`. This enables:

- **Cross-session tracking** — Concept mastery and trajectory plans persist across assessments
- **Personalized vocabulary** — Trajectory targets influence which alternatives are prioritized in upgrades
- **Backend compatibility** — Same DB used by the FastAPI backend when `ONTOLOGY_TRAJECTORY_ENABLED=1`

## Requirements

- **Permission**: Skill needs `all` (or filesystem) permission for SQLite read/write
- **Workspace**: Scripts must run from project root (`cd /path/to/SpeakingCoachV1`)
- **Backend**: `backend/` module must be importable (trajectory planner, vocabulary ontology, persistence). Use the project venv (e.g. `.venv-mlx/bin/python`) if available. If import fails (e.g. missing deps, MLX crash in headless env), scripts return empty trajectory_targets and exit 0.

## Scripts

| Script | When | Purpose |
|--------|------|---------|
| `scripts/load_trajectory.py` | Pre-assessment | Load state, output trajectory_targets for prompt injection |
| `scripts/update_trajectory.py` | Post-assessment | Update state with breakdown + recommendations, plan next trajectory |

## Usage

**Load** (before assessment):

```bash
cd /path/to/SpeakingCoachV1 && \
ONTOLOGY_TRAJECTORY_ENABLED=1 python .cursor/skills/ielts-speaking-coach/scripts/load_trajectory.py --user-id default
```

Output JSON includes `trajectory_targets`, `current_step`, `target_concepts`, `concept_mastery_summary`, `overall_band`.

**Update** (after assessment):

```bash
cd /path/to/SpeakingCoachV1 && \
ONTOLOGY_TRAJECTORY_ENABLED=1 python .cursor/skills/ielts-speaking-coach/scripts/update_trajectory.py \
  --user-id default --part 1 --json-file /tmp/assessment.json
```

Input JSON: `{"text": "...", "part": 1, "breakdown": {...}, "recommendations": [...]}`.

## Limitations

- **Concurrency**: If Backend and Skill both run, SQLite WAL allows concurrent reads; writes are serialized. Avoid heavy concurrent writes.
- **User ID**: Default `"default"` for single-user Cursor. Set `user_id` in skill config for multi-user.
- **Fallback**: If DB is missing or import fails, load returns empty trajectory_targets; update exits silently. Assessment proceeds without persistence.

## Schema

Uses [backend/persistence.py](backend/persistence.py) schema: `learner_states`, `concept_mastery`, `trajectory_plans`, `session_logs`.
