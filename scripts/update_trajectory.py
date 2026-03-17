#!/usr/bin/env python3
"""
Update learner state after assessment for skill-side persistence.

Usage (run from project root):
    cd /path/to/SpeakingCoachV1 && \
    ONTOLOGY_TRAJECTORY_ENABLED=1 python .cursor/skills/ielts-speaking-coach/scripts/update_trajectory.py \
        --user-id default --part 1 --json-file /tmp/assessment_payload.json

Or with stdin:
    echo '{"text":"...","part":1,"breakdown":{...},"recommendations":[...]}' | \
    ONTOLOGY_TRAJECTORY_ENABLED=1 python .../update_trajectory.py --user-id default --part 1
"""

import argparse
import json
import os
import signal
import sys

IMPORT_TIMEOUT_SEC = 10


def _workspace_root() -> str:
    """Project root (SpeakingCoachV1) from script location."""
    path = os.path.abspath(__file__)
    for _ in range(5):
        path = os.path.dirname(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Update learner trajectory after assessment")
    parser.add_argument("--user-id", default="default", help="Learner ID")
    parser.add_argument("--part", type=int, default=1, help="IELTS part (1, 2, or 3)")
    parser.add_argument("--json-file", help="Path to JSON payload; else read from stdin")
    args = parser.parse_args()
    user_id = (args.user_id or "default").strip() or "default"
    part = max(1, min(3, int(args.part)))

    if args.json_file:
        try:
            with open(args.json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"Error reading {args.json_file}: {e}", file=sys.stderr)
            return 1
    else:
        try:
            payload = json.load(sys.stdin)
        except Exception as e:
            print(f"Error reading stdin: {e}", file=sys.stderr)
            return 1

    text = str(payload.get("text", ""))
    raw_breakdown = payload.get("breakdown") or {}
    # Normalize keys: FC/LR/GRA/PR -> fluency/lexical/grammar/pronunciation
    key_map = {
        "FC": "fluency", "fluency": "fluency",
        "LR": "lexical", "lexical": "lexical",
        "GRA": "grammar", "grammar": "grammar",
        "PR": "pronunciation", "pronunciation": "pronunciation",
    }
    breakdown = {}
    for k, v in raw_breakdown.items():
        norm = key_map.get(k, k)
        if norm in ("fluency", "lexical", "grammar", "pronunciation"):
            try:
                breakdown[norm] = float(v)
            except (TypeError, ValueError):
                pass
    recommendations = payload.get("recommendations") or []

    # Ensure backend is importable
    root = _workspace_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    os.environ.setdefault("ONTOLOGY_TRAJECTORY_ENABLED", "1")
    db_path = os.path.join(root, "backend", "data", "learner_trajectory.db")
    os.environ["TRAJECTORY_DB_PATH"] = db_path

    # Lightweight pre-check: backend.persistence has no heavy deps
    try:
        import backend.persistence  # noqa: F401
    except ImportError as e:
        print(f"Backend not available: {e}", file=sys.stderr)
        return 0

    def _import_main():
        import backend.main as m
        return m

    try:
        if hasattr(signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Backend import timed out ({IMPORT_TIMEOUT_SEC}s)")

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(IMPORT_TIMEOUT_SEC)
        try:
            main_mod = _import_main()
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
    except (TimeoutError, ImportError) as e:
        print(f"Backend not available: {e}", file=sys.stderr)
        return 0

    try:
        main_mod.TRAJECTORY_DB_PATH = db_path
        main_mod.init_ontology_trajectory_system()

        if not main_mod.ONTOLOGY_TRAJECTORY_ENABLED or main_mod.STATE_STORE is None:
            return 0

        state = main_mod.get_learner_state(user_id)
        if state is None:
            return 0

        # Plan trajectory (same logic as get_smart_recommendations)
        if (
            main_mod.VOCAB_ONTOLOGY is not None
            and main_mod.TRAJECTORY_PLANNER is not None
        ):
            detected_concepts = main_mod.VOCAB_ONTOLOGY.map_text_to_concepts(
                text, part=part
            )
            weak_areas = main_mod._extract_weak_areas_from_breakdown(breakdown)
            session_signal = {
                "detected_concepts": detected_concepts,
                "weak_areas": weak_areas,
                "register_hint": "neutral",
            }
            target_band = float(state.overall_band()) + 0.5
            target_band = max(4.0, min(9.0, target_band))
            steps = main_mod.TRAJECTORY_PLANNER.plan_trajectory(
                state=state,
                session_signal=session_signal,
                target_band=target_band,
                min_diff=2.0,
                max_diff=4.0,
            )
            state.trajectory_plan = [s.to_dict() for s in steps]
            if state.trajectory_step >= len(state.trajectory_plan):
                state.trajectory_step = 0

            main_mod.persist_learner_state(user_id)
            if main_mod.STATE_STORE is not None:
                try:
                    main_mod.STATE_STORE.save_trajectory(
                        user_id, state.trajectory_plan
                    )
                except Exception:
                    pass

        # Update state with breakdown and recommendations
        main_mod.update_session_remap(
            user_id, text, part, breakdown, recommendations
        )

    except Exception as e:
        print(f"update_trajectory failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
