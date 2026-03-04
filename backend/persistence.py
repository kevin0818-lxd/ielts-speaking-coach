import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LearnerState:
    user_id: str
    domain_bands: Dict[str, float] = field(
        default_factory=lambda: {
            "fluency": 5.0,
            "lexical": 5.0,
            "grammar": 5.0,
            "pronunciation": 5.0,
        }
    )
    register_score: float = 5.0
    discourse_score: float = 5.0
    grammar_complexity: float = 5.0
    fluency: float = 5.0
    vocabulary_nodes: List[str] = field(default_factory=list)
    mastered_nodes: List[str] = field(default_factory=list)
    concept_mastery: Dict[str, float] = field(default_factory=dict)
    trajectory_step: int = 0
    trajectory_plan: List[Dict[str, Any]] = field(default_factory=list)
    target_profile: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def overall_band(self) -> float:
        vals = [
            float(self.domain_bands.get("fluency", self.fluency)),
            float(self.domain_bands.get("lexical", 5.0)),
            float(self.domain_bands.get("grammar", self.grammar_complexity)),
            float(self.domain_bands.get("pronunciation", 5.0)),
        ]
        avg = sum(vals) / 4.0 if vals else 5.0
        avg = max(1.0, min(9.0, avg))
        return round(avg * 2) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "domain_bands": self.domain_bands,
            "register_score": self.register_score,
            "discourse_score": self.discourse_score,
            "grammar_complexity": self.grammar_complexity,
            "fluency": self.fluency,
            "vocabulary_nodes": self.vocabulary_nodes,
            "mastered_nodes": self.mastered_nodes,
            "concept_mastery": self.concept_mastery,
            "trajectory_step": self.trajectory_step,
            "trajectory_plan": self.trajectory_plan,
            "target_profile": self.target_profile,
            "last_updated": self.last_updated,
            "overall_band": self.overall_band(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnerState":
        state = cls(user_id=str(data.get("user_id", "default")))
        state.domain_bands = dict(data.get("domain_bands", state.domain_bands))
        state.register_score = float(data.get("register_score", state.register_score))
        state.discourse_score = float(data.get("discourse_score", state.discourse_score))
        state.grammar_complexity = float(data.get("grammar_complexity", state.grammar_complexity))
        state.fluency = float(data.get("fluency", state.fluency))
        state.vocabulary_nodes = list(data.get("vocabulary_nodes", []))
        state.mastered_nodes = list(data.get("mastered_nodes", []))
        state.concept_mastery = dict(data.get("concept_mastery", {}))
        state.trajectory_step = int(data.get("trajectory_step", 0))
        state.trajectory_plan = list(data.get("trajectory_plan", []))
        state.target_profile = dict(data.get("target_profile", {}))
        state.last_updated = str(data.get("last_updated", state.last_updated))
        return state


class SQLitePersistence:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learner_states (
                    user_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectory_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    plan_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_mastery (
                    user_id TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    mastery REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, concept)
                )
                """
            )

    def load_state(self, user_id: str) -> LearnerState:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM learner_states WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return LearnerState(user_id=user_id)
            try:
                payload = json.loads(row["state_json"])
            except Exception:
                return LearnerState(user_id=user_id)
            return LearnerState.from_dict(payload)

    def save_state(self, user_id: str, state: LearnerState) -> None:
        state.last_updated = datetime.utcnow().isoformat()
        payload = json.dumps(state.to_dict(), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learner_states (user_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, payload, state.last_updated),
            )
            for concept, mastery in state.concept_mastery.items():
                conn.execute(
                    """
                    INSERT INTO concept_mastery (user_id, concept, mastery, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, concept) DO UPDATE SET
                        mastery = excluded.mastery,
                        updated_at = excluded.updated_at
                    """,
                    (user_id, concept, float(mastery), state.last_updated),
                )

    def append_session_log(self, user_id: str, payload: Dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat()
        body = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_logs (user_id, created_at, payload_json) VALUES (?, ?, ?)",
                (user_id, now, body),
            )

    def save_trajectory(self, user_id: str, plan: List[Dict[str, Any]]) -> None:
        now = datetime.utcnow().isoformat()
        plan_json = json.dumps(plan, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trajectory_plans (user_id, created_at, plan_json) VALUES (?, ?, ?)",
                (user_id, now, plan_json),
            )

    def load_latest_trajectory(self, user_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT plan_json FROM trajectory_plans
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if not row:
                return []
            try:
                value = json.loads(row["plan_json"])
            except Exception:
                return []
            return value if isinstance(value, list) else []
