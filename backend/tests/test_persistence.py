import os
import tempfile
import unittest

try:
    from backend.persistence import LearnerState, SQLitePersistence
except ImportError:
    from persistence import LearnerState, SQLitePersistence


class PersistenceTests(unittest.TestCase):
    def test_round_trip_state(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            store = SQLitePersistence(db_path)
            state = LearnerState(user_id="u1")
            state.concept_mastery["solid"] = 0.6
            state.trajectory_plan = [{"concept": "solid", "priority": 0.8}]
            store.save_state("u1", state)
            loaded = store.load_state("u1")
            self.assertEqual(loaded.user_id, "u1")
            self.assertIn("solid", loaded.concept_mastery)
            self.assertTrue(loaded.trajectory_plan)

    def test_user_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            store = SQLitePersistence(db_path)
            s1 = LearnerState(user_id="alice")
            s2 = LearnerState(user_id="bob")
            s1.concept_mastery["topic_a"] = 0.9
            s2.concept_mastery["topic_b"] = 0.2
            store.save_state("alice", s1)
            store.save_state("bob", s2)
            a = store.load_state("alice")
            b = store.load_state("bob")
            self.assertIn("topic_a", a.concept_mastery)
            self.assertNotIn("topic_a", b.concept_mastery)
            self.assertIn("topic_b", b.concept_mastery)


if __name__ == "__main__":
    unittest.main()
