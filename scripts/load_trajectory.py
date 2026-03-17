#!/usr/bin/env python3
"""
Load learner state and trajectory targets for skill-side persistence.

Usage (run from project root):
    cd /path/to/SpeakingCoachV1 && \
    ONTOLOGY_TRAJECTORY_ENABLED=1 python .cursor/skills/ielts-speaking-coach/scripts/load_trajectory.py --user-id default

Output: JSON to stdout with trajectory_targets, current_step, etc.
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
    parser = argparse.ArgumentParser(description="Load learner trajectory for skill")
    parser.add_argument("--user-id", default="default", help="Learner ID")
    args = parser.parse_args()
    user_id = (args.user_id or "default").strip() or "default"

    # Ensure backend is importable
    root = _workspace_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    os.environ.setdefault("ONTOLOGY_TRAJECTORY_ENABLED", "1")
    db_path = os.path.join(root, "backend", "data", "learner_trajectory.db")
    os.environ["TRAJECTORY_DB_PATH"] = db_path

    result = {
        "trajectory_targets": [],
        "current_step": None,
        "target_concepts": [],
        "concept_mastery_summary": {},
        "overall_band": 5.0,
    }

    # Lightweight pre-check: backend.persistence has no heavy deps
    try:
        import backend.persistence  # noqa: F401
    except ImportError as e:
        result["error"] = str(e)
        print(json.dumps(result, ensure_ascii=False))
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
        result["error"] = str(e)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    try:
        main_mod.TRAJECTORY_DB_PATH = db_path
        main_mod.init_ontology_trajectory_system()

        if not main_mod.ONTOLOGY_TRAJECTORY_ENABLED or main_mod.STATE_STORE is None:
            print(json.dumps(result, ensure_ascii=False))
            return 0

        state = main_mod.get_learner_state(user_id)
        if state is None:
            print(json.dumps(result, ensure_ascii=False))
            return 0

        result["overall_band"] = float(state.overall_band())
        result["concept_mastery_summary"] = dict(state.concept_mastery)

        if (
            main_mod.VOCAB_ONTOLOGY is not None
            and state.trajectory_plan
            and 0 <= state.trajectory_step < len(state.trajectory_plan)
        ):
            current_step = state.trajectory_plan[state.trajectory_step]
            result["current_step"] = current_step
            result["target_concepts"] = [
                str(s.get("concept", "")).strip()
                for s in state.trajectory_plan[:3]
                if s.get("concept")
            ]

            trajectory_targets = set()
            current_concept = str(current_step.get("concept", "")).strip().lower()
            if current_concept:
                for n in main_mod.VOCAB_ONTOLOGY.get_candidate_neighbors(
                    [current_concept], max_hops=1, register_hint="neutral"
                ):
                    trajectory_targets.add(str(n).lower())
                for pre in main_mod.VOCAB_ONTOLOGY.get_prerequisite_chain(current_concept):
                    trajectory_targets.add(str(pre).lower())
            for step in state.trajectory_plan[:4]:
                concept = str(step.get("concept", "")).strip().lower()
                if concept:
                    trajectory_targets.add(concept)

            result["trajectory_targets"] = sorted(trajectory_targets)

    except Exception as e:
        result["error"] = str(e)
        result["trajectory_targets"] = []

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
