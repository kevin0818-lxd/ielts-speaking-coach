from dataclasses import dataclass
from typing import Any, Dict, List

try:
    from backend.persistence import LearnerState
except ImportError:
    from persistence import LearnerState


@dataclass
class TrajectoryStep:
    concept: str
    rationale: str
    priority: float
    source: str = "ontology"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "rationale": self.rationale,
            "priority": round(float(self.priority), 3),
            "source": self.source,
        }


class TrajectoryPlanner:
    """Selects next concept targets from learner state + session signal."""

    def __init__(self, ontology) -> None:
        self.ontology = ontology

    def plan_trajectory(
        self,
        state: LearnerState,
        session_signal: Dict[str, Any],
        target_band: float,
        min_diff: float = 2.0,
        max_diff: float = 4.0,
    ) -> List[TrajectoryStep]:
        concepts = list(session_signal.get("detected_concepts", []))
        weaknesses = list(session_signal.get("weak_areas", []))
        if not concepts:
            concepts = ["general"]

        register_hint = session_signal.get("register_hint", "neutral")
        neighbors = self.ontology.get_candidate_neighbors(
            concepts=concepts,
            max_hops=2,
            register_hint=register_hint,
        )

        # Score candidate concepts/words by mastery gap + weakness alignment.
        scored: List[TrajectoryStep] = []
        for cand in neighbors:
            mastery = float(state.concept_mastery.get(cand, 0.0))
            gap = 1.0 - max(0.0, min(1.0, mastery))
            weak_bonus = 0.0
            for area in weaknesses:
                if str(area).lower() in str(cand).lower():
                    weak_bonus += 0.15
            prereq_chain = self.ontology.get_prerequisite_chain(cand)
            prereq_penalty = 0.08 * len(prereq_chain)
            priority = gap + weak_bonus - prereq_penalty
            if priority <= 0.05:
                continue
            rationale = "Close lexical gap"
            if weak_bonus > 0:
                rationale = "Targets detected weak area"
            if prereq_chain:
                rationale += f"; prerequisites: {', '.join(prereq_chain[-2:])}"
            scored.append(TrajectoryStep(concept=cand, rationale=rationale, priority=priority))

        if not scored:
            scored.append(
                TrajectoryStep(
                    concept="coherence markers",
                    rationale="Fallback practice target",
                    priority=0.3,
                    source="fallback",
                )
            )

        scored.sort(key=lambda s: s.priority, reverse=True)
        top_steps = scored[:5]
        return top_steps
