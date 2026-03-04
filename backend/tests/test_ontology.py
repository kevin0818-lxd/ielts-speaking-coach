import unittest

try:
    from backend.vocabulary_ontology import VocabularyOntology
except ImportError:
    from vocabulary_ontology import VocabularyOntology


class OntologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context_vocab = {
            "good": {
                "default": [{"word": "solid", "reg": "spoken", "diff": 2}],
                "food": [{"word": "delicious", "reg": "spoken", "diff": 2}],
            }
        }
        self.context_keywords = {
            "food": ["eat", "meal", "delicious"],
            "work": ["job", "office"],
        }
        self.ontology = VocabularyOntology(self.context_vocab, self.context_keywords, seed_path=None)

    def test_map_text_to_concepts(self):
        concepts = self.ontology.map_text_to_concepts("I had a delicious meal yesterday", part=1)
        self.assertIn("food", concepts)

    def test_neighbor_lookup(self):
        recs = self.ontology.get_candidate_neighbors(["food"], max_hops=2, register_hint="spoken")
        self.assertTrue(any(word in recs for word in ("delicious", "solid")))


if __name__ == "__main__":
    unittest.main()
