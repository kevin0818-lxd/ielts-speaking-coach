import unittest

try:
    from backend.persistence import LearnerState
    from backend.trajectory_planner import TrajectoryPlanner
    from backend.vocabulary_ontology import VocabularyOntology
except ImportError:
    from persistence import LearnerState
    from trajectory_planner import TrajectoryPlanner
    from vocabulary_ontology import VocabularyOntology


class PlannerTests(unittest.TestCase):
    def test_prerequisite_aware_plan(self):
        context_vocab = {
            "good": {
                "default": [{"word": "decent", "reg": "spoken", "diff": 2}],
                "work": [{"word": "productive", "reg": "neutral", "diff": 3}],
            }
        }
        context_keywords = {"work": ["job", "office", "career"]}
        ontology = VocabularyOntology(context_vocab, context_keywords, seed_path=None)
        planner = TrajectoryPlanner(ontology)
        state = LearnerState(user_id="u1")
        state.concept_mastery["decent"] = 0.1
        steps = planner.plan_trajectory(
            state=state,
            session_signal={"detected_concepts": ["work"], "weak_areas": ["lexical"], "register_hint": "neutral"},
            target_band=7.0,
        )
        self.assertTrue(steps)
        self.assertIsInstance(steps[0].concept, str)


if __name__ == "__main__":
    unittest.main()
