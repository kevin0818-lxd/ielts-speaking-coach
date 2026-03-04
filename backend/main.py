from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import re
import os
import asyncio
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import math
# import whisper # Removed to save memory, using subprocess
import torch
import librosa
import soundfile as sf
import shutil
import tempfile
import subprocess
import json
import sys
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LLM_MERGE_FIX_VERSION = "1.0.0"
from gtts import gTTS
from io import BytesIO

# Load environment variables
load_dotenv()

# ============================================================================
# AI-DRIVEN PEDAGOGICAL WORD RECOMMENDATION SYSTEM
# Following the workflow from: "AI-Driven Pedagogical Word Recommendation 
# Systems Transforming English as a Second Language Vocabulary Learning Effectiveness"
# IEEE ICMLAS 2025
# ============================================================================

# ============================================================================
# USER PROFILE SYSTEM - Records user skills, objectives, and preferences
# (Figure 4: Step 1 - User Profile Creation)
# ============================================================================

class UserProfile:
    """
    Learner profile for personalized vocabulary recommendations.
    Tracks proficiency level, learning goals, and preferences.
    """
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.proficiency_level = 5.0  # IELTS band score estimate (1-9)
        self.target_level = 7.0  # Target band score
        self.learning_goals = []  # e.g., ["academic", "conversational", "professional"]
        self.preferred_topics = []  # e.g., ["technology", "environment", "education"]
        self.register_preference = "neutral"  # "formal", "neutral", "spoken"
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.total_sessions = 0
        self.vocabulary_mastered = set()  # Words the user has demonstrated mastery
        self.vocabulary_learning = set()  # Words currently being learned
        self.vocabulary_introduced = set()  # Words introduced but not yet learned
        
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "proficiency_level": self.proficiency_level,
            "target_level": self.target_level,
            "learning_goals": self.learning_goals,
            "preferred_topics": self.preferred_topics,
            "register_preference": self.register_preference,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "total_sessions": self.total_sessions,
            "vocabulary_mastered": list(self.vocabulary_mastered),
            "vocabulary_learning": list(self.vocabulary_learning),
            "vocabulary_introduced": list(self.vocabulary_introduced),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        profile = cls(data.get("user_id", "default"))
        profile.proficiency_level = data.get("proficiency_level", 5.0)
        profile.target_level = data.get("target_level", 7.0)
        profile.learning_goals = data.get("learning_goals", [])
        profile.preferred_topics = data.get("preferred_topics", [])
        profile.register_preference = data.get("register_preference", "neutral")
        profile.total_sessions = data.get("total_sessions", 0)
        profile.vocabulary_mastered = set(data.get("vocabulary_mastered", []))
        profile.vocabulary_learning = set(data.get("vocabulary_learning", []))
        profile.vocabulary_introduced = set(data.get("vocabulary_introduced", []))
        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if "last_active" in data:
            profile.last_active = datetime.fromisoformat(data["last_active"])
        return profile


# ============================================================================
# LEARNING REPOSITORY - Performance logging and retention tracking
# (Figure 4: Step 5 - Performance Logged in Learning Repository)
# ============================================================================

class LearningRepository:
    """
    Stores all learning interactions for AI analytics and model refinement.
    Tracks vocabulary exposure, retention, and engagement metrics.
    """
    def __init__(self):
        # Word-level tracking
        self.word_interactions: Dict[str, List[Dict]] = defaultdict(list)
        # Session-level tracking  
        self.sessions: List[Dict] = []
        # Retention scores (word -> retention score 0-1)
        self.retention_scores: Dict[str, float] = {}
        # Engagement metrics per word
        self.engagement_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "shown_count": 0,
            "accepted_count": 0,
            "ignored_count": 0,
            "used_in_speech_count": 0,
            "time_to_accept_ms": [],
            "last_interaction": None,
            "first_seen": None,
            "mastery_level": 0  # 0=new, 1=introduced, 2=learning, 3=learned, 4=mastered
        })
        # Proficiency tracking over time
        self.proficiency_history: List[Dict] = []
        
    def log_interaction(self, word: str, action: str, context: str = "", 
                       response_time_ms: float = 0, metadata: Dict = None):
        """Log a vocabulary interaction event."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "action": action,  # "shown", "accepted", "ignored", "used", "tested"
            "context": context,
            "response_time_ms": response_time_ms,
            "metadata": metadata or {}
        }
        self.word_interactions[word.lower()].append(interaction)
        
        # Update engagement metrics
        metrics = self.engagement_metrics[word.lower()]
        if metrics["first_seen"] is None:
            metrics["first_seen"] = interaction["timestamp"]
        metrics["last_interaction"] = interaction["timestamp"]
        
        if action == "shown":
            metrics["shown_count"] += 1
        elif action == "accepted":
            metrics["accepted_count"] += 1
            if response_time_ms > 0:
                metrics["time_to_accept_ms"].append(response_time_ms)
            self._update_mastery_level(word)
        elif action == "ignored":
            metrics["ignored_count"] += 1
        elif action == "used":
            metrics["used_in_speech_count"] += 1
            self._update_mastery_level(word)
            
    def _update_mastery_level(self, word: str):
        """Update word mastery level based on interactions."""
        metrics = self.engagement_metrics[word.lower()]
        accepted = metrics["accepted_count"]
        used = metrics["used_in_speech_count"]
        
        # Mastery progression: 0=new, 1=introduced, 2=learning, 3=learned, 4=mastered
        if used >= 3 and accepted >= 2:
            metrics["mastery_level"] = 4  # Mastered
        elif used >= 2 or accepted >= 3:
            metrics["mastery_level"] = 3  # Learned
        elif accepted >= 1 or used >= 1:
            metrics["mastery_level"] = 2  # Learning
        elif metrics["shown_count"] >= 1:
            metrics["mastery_level"] = 1  # Introduced
            
    def calculate_retention_score(self, word: str) -> float:
        """
        Calculate retention score using spaced repetition principles.
        Based on Ebbinghaus forgetting curve.
        """
        metrics = self.engagement_metrics[word.lower()]
        if metrics["first_seen"] is None:
            return 0.0
            
        # Factors for retention
        accepted = metrics["accepted_count"]
        used = metrics["used_in_speech_count"]
        mastery = metrics["mastery_level"]
        
        # Time since last interaction (decay factor)
        last_interaction = metrics["last_interaction"]
        if last_interaction:
            try:
                last_dt = datetime.fromisoformat(last_interaction)
                hours_since = (datetime.now() - last_dt).total_seconds() / 3600
                # Forgetting curve: retention = e^(-time/strength)
                strength = 24 * (1 + accepted + used * 2)  # Base 24 hours, increased by practice
                decay = math.exp(-hours_since / strength)
            except:
                decay = 0.5
        else:
            decay = 0.5
            
        # Base retention from mastery level
        base_retention = mastery / 4.0
        
        # Combined score
        retention = min(1.0, base_retention * 0.6 + decay * 0.4)
        self.retention_scores[word.lower()] = retention
        return retention
        
    def get_word_stats(self, word: str) -> Dict:
        """Get comprehensive statistics for a word."""
        metrics = self.engagement_metrics[word.lower()]
        retention = self.calculate_retention_score(word)
        return {
            **metrics,
            "retention_score": retention,
            "interaction_count": len(self.word_interactions.get(word.lower(), []))
        }
        
    def log_session(self, session_data: Dict):
        """Log a complete learning session."""
        session_data["timestamp"] = datetime.now().isoformat()
        self.sessions.append(session_data)
        
    def log_proficiency_update(self, score: float, component_scores: Dict = None):
        """Track proficiency changes over time."""
        self.proficiency_history.append({
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "component_scores": component_scores or {}
        })


# ============================================================================
# AI SCORING SYSTEM - Intelligent Word Ranking and Selection
# (Figure 2: AI Scoring System -> Ranked Word List)
# ============================================================================

class AIWordScoringSystem:
    """
    Ranks and selects vocabulary based on difficulty, frequency, 
    contextual relevance, and learner competency.
    """
    
    # Word difficulty levels based on CEFR/frequency
    DIFFICULTY_LEVELS = {
        1: "A1-A2 (Basic)",
        2: "A2-B1 (Elementary)", 
        3: "B1-B2 (Intermediate)",
        4: "B2-C1 (Upper Intermediate)",
        5: "C1-C2 (Advanced)"
    }
    
    def __init__(self, user_profile: UserProfile, learning_repo: LearningRepository):
        self.user_profile = user_profile
        self.learning_repo = learning_repo
        
    def calculate_word_score(self, word: str, candidate: Dict, 
                            context_similarity: float = 0.0,
                            is_context_match: bool = False) -> float:
        """
        Calculate comprehensive score for a word recommendation.
        
        Scoring factors:
        1. Difficulty appropriateness (i+1 principle - slightly above current level)
        2. Contextual relevance
        3. Register appropriateness
        4. Retention status (don't recommend mastered words)
        5. Engagement history (avoid repeatedly ignored words)
        6. Learning goal alignment
        """
        score = 0.0
        
        # 1. Difficulty appropriateness (Krashen's i+1 hypothesis)
        word_diff = candidate.get("diff", 3)
        user_level = self.user_profile.proficiency_level
        
        # Map IELTS band to difficulty level (1-5)
        user_diff_level = max(1, min(5, (user_level - 3) / 1.2 + 1))
        
        # Optimal: word is 0.5-1.5 levels above current
        diff_delta = word_diff - user_diff_level
        if 0.5 <= diff_delta <= 1.5:
            score += 15  # Optimal challenge zone
        elif 0 <= diff_delta <= 2:
            score += 10  # Acceptable range
        elif diff_delta < 0:
            score += max(0, 5 + diff_delta * 2)  # Too easy
        else:
            score += max(0, 8 - (diff_delta - 2) * 3)  # Too hard
            
        # 2. Contextual relevance
        if is_context_match:
            score += 12  # Strong context match bonus
        score += context_similarity * 8  # Semantic similarity bonus
        
        # 3. Register appropriateness
        reg = candidate.get("reg", "neutral")
        pref = self.user_profile.register_preference
        
        reg_scores = {
            ("formal", "formal"): 5,
            ("formal", "neutral"): 3,
            ("neutral", "neutral"): 4,
            ("neutral", "formal"): 3,
            ("neutral", "spoken"): 3,
            ("spoken", "spoken"): 5,
            ("spoken", "neutral"): 3,
            ("academic", "formal"): 4,
            ("academic", "neutral"): 2,
        }
        score += reg_scores.get((reg, pref), 2)
        
        # 4. Retention status - avoid recommending mastered words
        word_key = word.lower() if isinstance(word, str) else str(candidate.get("word", "")).lower()
        stats = self.learning_repo.get_word_stats(word_key)
        mastery = stats.get("mastery_level", 0)
        retention = stats.get("retention_score", 0)
        
        if mastery >= 4:  # Mastered
            score -= 20  # Strong penalty
        elif mastery >= 3:  # Learned
            score -= 10
        elif mastery >= 2:  # Learning - reinforce
            score += 5 * (1 - retention)  # Bonus if retention dropping
            
        # 5. Engagement history - avoid ignored words
        ignored = stats.get("ignored_count", 0)
        accepted = stats.get("accepted_count", 0)
        
        if ignored > 0:
            score -= 15 * ignored  # Strong penalty for ignored words
        if accepted > 0:
            # Reduce score for previously accepted (variety)
            score -= 3 * min(accepted, 3)
            
        # 6. Learning goal alignment
        word_topics = candidate.get("topics", [])
        word_topics = [t for t in word_topics if isinstance(t, str)]
        learning_goals = [g for g in (self.user_profile.learning_goals or []) if isinstance(g, str)]
        goal_match = len(set(word_topics) & set(learning_goals))
        score += goal_match * 3
        
        # 7. Time-based decay for recently shown words
        last_interaction = stats.get("last_interaction")
        if last_interaction:
            try:
                last_dt = datetime.fromisoformat(last_interaction)
                minutes_since = (datetime.now() - last_dt).total_seconds() / 60
                if minutes_since < 5:
                    score -= 10  # Very recent - avoid repetition
                elif minutes_since < 30:
                    score -= 5
            except:
                pass
                
        return score
        
    def rank_candidates(self, candidates: List[Dict], 
                       context_text: str = "") -> List[tuple]:
        """
        Rank all candidates and return sorted list of (score, candidate).
        """
        scored = []
        for cand in candidates:
            word = cand.get("word", "")
            context_sim = cand.get("context_similarity", 0.0)
            is_context = cand.get("source") == "context"
            
            score = self.calculate_word_score(
                word, cand, 
                context_similarity=context_sim,
                is_context_match=is_context
            )
            scored.append((score, cand))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


# ============================================================================
# ADAPTIVE LEARNING ENGINE - Dynamic difficulty adjustment
# (Figure 2: Adaptive Learning Engine)
# ============================================================================

class AdaptiveLearningEngine:
    """
    Dynamically adjusts instructional material and word recommendations
    based on learner performance and engagement patterns.
    """
    
    def __init__(self, user_profile: UserProfile, learning_repo: LearningRepository):
        self.user_profile = user_profile
        self.learning_repo = learning_repo
        
    def analyze_user_performance(self) -> Dict:
        """
        Analyze user's learning performance from repository data.
        """
        metrics = self.learning_repo.engagement_metrics
        
        total_shown = sum(m["shown_count"] for m in metrics.values())
        total_accepted = sum(m["accepted_count"] for m in metrics.values())
        total_used = sum(m["used_in_speech_count"] for m in metrics.values())
        
        # Calculate acceptance rate
        acceptance_rate = total_accepted / max(1, total_shown)
        
        # Calculate usage rate (actually using recommended words)
        usage_rate = total_used / max(1, total_accepted)
        
        # Mastery distribution
        mastery_dist = defaultdict(int)
        for m in metrics.values():
            mastery_dist[m["mastery_level"]] += 1
            
        # Average retention across all words
        retentions = [self.learning_repo.calculate_retention_score(w) 
                     for w in metrics.keys()]
        avg_retention = sum(retentions) / max(1, len(retentions))
        
        return {
            "total_words_exposed": len(metrics),
            "total_interactions": total_shown,
            "acceptance_rate": acceptance_rate,
            "usage_rate": usage_rate,
            "mastery_distribution": dict(mastery_dist),
            "average_retention": avg_retention
        }
        
    def should_adjust_difficulty(self) -> tuple:
        """
        Determine if difficulty level should be adjusted.
        Returns (should_adjust: bool, direction: str, magnitude: float)
        """
        perf = self.analyze_user_performance()
        
        # Not enough data yet
        if perf["total_interactions"] < 10:
            return (False, "none", 0)
            
        acceptance_rate = perf["acceptance_rate"]
        usage_rate = perf["usage_rate"]
        avg_retention = perf["average_retention"]
        
        # High acceptance + high usage + high retention = increase difficulty
        if acceptance_rate > 0.7 and usage_rate > 0.5 and avg_retention > 0.7:
            magnitude = min(0.5, (acceptance_rate - 0.5) * 0.5)
            return (True, "increase", magnitude)
            
        # Low acceptance or very low retention = decrease difficulty
        if acceptance_rate < 0.3 or avg_retention < 0.3:
            magnitude = min(0.5, (0.5 - acceptance_rate) * 0.5)
            return (True, "decrease", magnitude)
            
        return (False, "none", 0)
        
    def update_proficiency_estimate(self, evaluation_scores: Dict = None):
        """
        Update user's estimated proficiency based on performance data.
        """
        perf = self.analyze_user_performance()
        current_level = self.user_profile.proficiency_level
        
        # Factor 1: Acceptance and usage patterns
        if perf["total_interactions"] >= 10:
            should_adjust, direction, magnitude = self.should_adjust_difficulty()
            if should_adjust:
                if direction == "increase":
                    current_level = min(9.0, current_level + magnitude)
                elif direction == "decrease":
                    current_level = max(1.0, current_level - magnitude)
                    
        # Factor 2: Evaluation scores if provided
        if evaluation_scores:
            avg_score = sum(evaluation_scores.values()) / max(1, len(evaluation_scores))
            # Weighted average with current estimate
            current_level = current_level * 0.7 + avg_score * 0.3
            
        self.user_profile.proficiency_level = round(current_level, 1)
        
        # Log proficiency update
        self.learning_repo.log_proficiency_update(
            current_level, 
            evaluation_scores
        )
        
        return current_level
        
    def get_recommended_difficulty_range(self) -> tuple:
        """
        Get the recommended word difficulty range for current user.
        Returns (min_diff, optimal_diff, max_diff)
        """
        level = self.user_profile.proficiency_level
        
        # Map IELTS band to difficulty (1-5)
        base_diff = max(1, min(5, (level - 3) / 1.2 + 1))
        
        # i+1 principle: recommend slightly above current level
        optimal = min(5, base_diff + 0.5)
        min_diff = max(1, base_diff - 0.5)
        max_diff = min(5, base_diff + 1.5)
        
        return (min_diff, optimal, max_diff)
        
    def select_words_for_reinforcement(self) -> List[str]:
        """
        Select words that need reinforcement based on spaced repetition.
        """
        words_to_reinforce = []
        
        for word, metrics in self.learning_repo.engagement_metrics.items():
            mastery = metrics["mastery_level"]
            retention = self.learning_repo.calculate_retention_score(word)
            
            # Words in learning phase (2) with dropping retention
            if mastery == 2 and retention < 0.5:
                words_to_reinforce.append((word, retention))
            # Learned words (3) that might be forgotten
            elif mastery == 3 and retention < 0.4:
                words_to_reinforce.append((word, retention))
                
        # Sort by retention (lowest first - most need reinforcement)
        words_to_reinforce.sort(key=lambda x: x[1])
        
        return [w for w, _ in words_to_reinforce[:5]]


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Global user profiles storage (in production, use database)
USER_PROFILES: Dict[str, UserProfile] = {}

# Global learning repository (in production, use database)
LEARNING_REPOSITORIES: Dict[str, LearningRepository] = {}

# Feature flag for ontology-driven personalization rollout.
ONTOLOGY_TRAJECTORY_ENABLED = os.environ.get("ONTOLOGY_TRAJECTORY_ENABLED", "0") == "1"
TRAJECTORY_DB_PATH = os.environ.get(
    "TRAJECTORY_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "learner_trajectory.db"),
)

LEARNER_STATES: Dict[str, Any] = {}
STATE_STORE = None
VOCAB_ONTOLOGY = None
TRAJECTORY_PLANNER = None

def get_user_profile(user_id: str = "default") -> UserProfile:
    """Get or create user profile."""
    if user_id not in USER_PROFILES:
        USER_PROFILES[user_id] = UserProfile(user_id)
    return USER_PROFILES[user_id]

def get_learning_repo(user_id: str = "default") -> LearningRepository:
    """Get or create learning repository."""
    if user_id not in LEARNING_REPOSITORIES:
        LEARNING_REPOSITORIES[user_id] = LearningRepository()
    return LEARNING_REPOSITORIES[user_id]


try:
    try:
        from backend.persistence import LearnerState, SQLitePersistence
    except ImportError:
        from persistence import LearnerState, SQLitePersistence
except Exception:
    LearnerState = None
    SQLitePersistence = None

try:
    try:
        from backend.vocabulary_ontology import VocabularyOntology
    except ImportError:
        from vocabulary_ontology import VocabularyOntology
except Exception:
    VocabularyOntology = None

try:
    try:
        from backend.trajectory_planner import TrajectoryPlanner
    except ImportError:
        from trajectory_planner import TrajectoryPlanner
except Exception:
    TrajectoryPlanner = None

# Import LLM Evaluator
try:
    try:
        from backend.llm_evaluator import get_llm_evaluator
    except ImportError:
        from llm_evaluator import get_llm_evaluator
except ImportError:
    # Fallback if file not found or path issue
    def get_llm_evaluator(): return None

try:
    from base_corpus import get_base_calibrator
except ImportError:
    def get_base_calibrator(): return None

_fcael_stats = None

def get_fcael_stats():
    global _fcael_stats
    if _fcael_stats is not None:
        return _fcael_stats
    path = os.path.join(os.path.dirname(__file__), "resources", "fcael_stats.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _fcael_stats = json.load(f)
    except Exception:
        _fcael_stats = {}
    return _fcael_stats

def compute_temporal_fluency_features(audio_path: str):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        total_dur = len(y) / sr if sr else 0.0
        if total_dur <= 0:
            return None
        intervals = librosa.effects.split(y, top_db=35)
        speech_dur = 0.0
        last_end = 0
        pause_durs = []
        for start, end in intervals:
            speech_dur += (end - start) / sr
            if start > last_end:
                pause_durs.append((start - last_end) / sr)
            last_end = end
        if last_end < len(y):
            pause_durs.append((len(y) - last_end) / sr)
        pause_dur = max(0.0, total_dur - speech_dur)
        pause_ratio = pause_dur / total_dur if total_dur else 0.0
        mean_pause_s = float(sum(pause_durs) / len(pause_durs)) if pause_durs else 0.0
        pause_count_per_min = (len(pause_durs) / total_dur) * 60 if total_dur else 0.0
        phonation_time_ratio = speech_dur / total_dur if total_dur else 0.0
        return {
            "total_dur_s": float(total_dur),
            "speech_dur_s": float(speech_dur),
            "pause_ratio": float(pause_ratio),
            "mean_pause_s": float(mean_pause_s),
            "pause_count_per_min": float(pause_count_per_min),
            "phonation_time_ratio": float(phonation_time_ratio),
        }
    except Exception:
        return None

def compute_mfcc_summary(audio_path: str):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if sr <= 0 or y is None or len(y) == 0:
            return None
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmin=50, fmax=8000)
        log_mels = librosa.power_to_db(mels, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mels, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mel_freqs = librosa.mel_frequencies(n_mels=40, fmin=50, fmax=8000)
        band_mask = (mel_freqs >= 300) & (mel_freqs <= 3400)
        band_energy = float(mels[band_mask, :].sum()) if band_mask.any() else 0.0
        total_energy = float(mels.sum()) if mels.size else 0.0
        band_ratio = float(band_energy / total_energy) if total_energy > 0 else 0.0
        return {
            "sr": int(sr),
            "n_mels": 40,
            "n_mfcc": 13,
            "fmin": 50,
            "fmax": 8000,
            "mfcc_mean": [float(x) for x in mfcc.mean(axis=1)],
            "mfcc_std": [float(x) for x in mfcc.std(axis=1)],
            "delta_mean": [float(x) for x in delta.mean(axis=1)],
            "delta_std": [float(x) for x in delta.std(axis=1)],
            "delta2_mean": [float(x) for x in delta2.mean(axis=1)],
            "delta2_std": [float(x) for x in delta2.std(axis=1)],
            "speech_band_energy_ratio": band_ratio,
        }
    except Exception:
        return None

def _get_local_model_path() -> str:
    """Resolve local model path without importing llm_evaluator (avoids peft dependency at startup)."""
    override = os.environ.get("LOCAL_MODEL_PATH")
    if override and os.path.exists(override):
        return override
    _dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_dir, "resources", "llm_finetune", "checkpoint-50"),
        os.path.join(os.getcwd(), "backend", "resources", "llm_finetune", "checkpoint-50"),
        os.path.join(os.getcwd(), "resources", "llm_finetune", "checkpoint-50"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up IELTS Speaking Coach Backend...")
    logger.info(f"LLM merge fix version: {LLM_MERGE_FIX_VERSION}")
    local_path = _get_local_model_path()
    logger.info(f"Local model path: {local_path} (exists={os.path.exists(local_path)})")
    try:
        ev = get_llm_evaluator()
        status = "loaded" if (ev and ev.is_available()) else "unavailable"
        logger.info(f"LLM Evaluator: {status}")
    except Exception as e:
        logger.warning(f"LLM Evaluator preload failed (install peft: pip install peft): {e}")
    try:
        init_ontology_trajectory_system()
    except Exception as e:
        logger.warning(f"Ontology trajectory init failed: {e}")
    yield
    # Shutdown
    pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"status": "ok", "service": "IELTS Speaking Coach API"}


@app.get("/health")
async def health_check():
    """Lightweight health check for connectivity verification (e.g. curl https://YOUR_DOMAIN/health)."""
    ontology_stats = {}
    if ONTOLOGY_TRAJECTORY_ENABLED and VOCAB_ONTOLOGY is not None:
        try:
            ontology_stats = VOCAB_ONTOLOGY.stats()
        except Exception:
            ontology_stats = {}
    return {
        "status": "ok",
        "service": "IELTS Speaking Coach",
        "ontology_trajectory_enabled": ONTOLOGY_TRAJECTORY_ENABLED,
        "ontology_stats": ontology_stats,
    }


import language_tool_python
from textblob import TextBlob
import spacy
import numpy as np

# Load spaCy model with word vectors
nlp = None # Lazy load

def get_spacy():
    global nlp
    if nlp:
        return nlp
    try:
        print("Lazy loading spaCy model...")
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Warning: en_core_web_md not found. Downloading...")
        from spacy.cli import download
        download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    return nlp

# Initialize grammar tool globally to avoid reloading
grammar_tool = None

def get_grammar_tool():
    global grammar_tool
    if grammar_tool:
        return grammar_tool
    try:
        print("Lazy loading LanguageTool...")
        grammar_tool = language_tool_python.LanguageTool('en-US')
    except Exception as e:
        print(f"Warning: Could not initialize LanguageTool: {e}")
        grammar_tool = None
    return grammar_tool

def _extract_languagetool_original(text: str, match) -> str:
    try:
        start = int(getattr(match, "offset", 0) or 0)
    except Exception:
        start = 0

    length = getattr(match, "errorLength", None)
    if length is None:
        length = getattr(match, "error_length", None)
    if length is None:
        length = getattr(match, "length", None)

    try:
        length = int(length) if length is not None else 0
    except Exception:
        length = 0

    if length <= 0:
        length = 1

    start = max(0, min(len(text), start))
    end = max(start, min(len(text), start + length))
    return text[start:end]

def _extract_languagetool_rule_id(match) -> str:
    rid = getattr(match, "ruleId", None)
    if rid is None:
        rid = getattr(match, "rule_id", None)
    if rid is None:
        rule = getattr(match, "rule", None)
        if rule is not None:
            rid = getattr(rule, "id", None)
    return str(rid) if rid is not None else ""

def _extract_languagetool_message(match) -> str:
    msg = getattr(match, "message", None)
    return str(msg) if msg is not None else ""

def _extract_languagetool_replacements(match) -> list:
    repls = getattr(match, "replacements", None)
    if repls is None:
        repls = getattr(match, "replacement", None)
    if repls is None:
        return []
    if isinstance(repls, (list, tuple)):
        return [str(r) for r in repls if r is not None]
    return [str(repls)]

# Initialize Whisper Model (ASR)
whisper_model = None # Lazy load

# Lexical Bundles Taxonomy (Based on Biber et al. 2004)
# Spoken registers (Conversation/Teaching) rely heavily on Stance and Discourse bundles.
# Written registers (Textbooks/Academic Prose) rely more on Referential bundles.
# For IELTS Speaking (Spoken Register), we reward Stance and Discourse bundles highly.

# 1. Stance Bundles (Epistemic, Attitudinal) - Vital for Spoken Fluency & Coherence
SPOKEN_STANCE_BUNDLES = [
    "I don't know", "I think it was", "I don't think", "I don't want to", "I want to",
    "I would say", "I mean", "you know", "it seems to me", "if you ask me",
    "I'm not sure", "I suppose", "I guess", "to be honest", "as far as I know",
    "it is likely that", "it is possible that", "I strongly believe", "I have no doubt",
    "what I want to do", "what I'm saying is", "the thing is", "to tell you the truth"
]

# 2. Discourse Organizing Bundles (Topic Introduction, Clarification) - Vital for Coherence
SPOKEN_DISCOURSE_BUNDLES = [
    "what I mean is", "let me see", "going to talk about", "have a look at",
    "as well as", "on the other hand", "first of all", "at the same time",
    "in other words", "what happens is", "the reason is", "as a result of",
    "moving on to", "that reminds me", "funny enough", "believe it or not",
    "speaking of which", "come to think of it"
]

# 3. Referential Bundles (Identification, Quantity, Time/Place) - Common in Written/Academic, less in Conversation
# Overuse in speaking might sound "bookish", but appropriate use is good for "Lexical Resource"
REFERENTIAL_BUNDLES = [
    "one of the", "part of the", "the end of the", "the number of", "a lot of",
    "at the end of", "the nature of the", "in the form of", "the result of the",
    "on the basis of", "in the case of", "a wide range of", "the fact that the",
    "as a result of", "in addition to the", "associated with the", "in terms of",
    "economic situation", "technological advancement", "bird's eye view"
]

# 4. Academic Spoken Word List (ASWL) - Dang, Coxhead, Webb (2017)
# Selected high-frequency academic spoken vocabulary for Part 3 (Abstract/Discussion)
ASWL_VOCAB = {
    # Level 1 (Very High Frequency in Academic Speech)
    "actual", "actually", "analysis", "approach", "area", "assessment", "available", 
    "based", "basic", "basis", "benefit", "category", "central", "change", "clear", 
    "community", "complete", "complex", "component", "concept", "condition", "context", 
    "create", "critical", "current", "data", "define", "definition", "design", "development", 
    "difference", "different", "discussion", "economic", "effect", "effective", "element", 
    "environment", "evidence", "example", "factor", "feature", "final", "focus", "form", 
    "function", "general", "global", "identify", "impact", "important", "include", "individual", 
    "information", "issue", "level", "major", "method", "model", "nature", "necessary", 
    "particular", "percent", "period", "policy", "positive", "potential", "practice", "process", 
    "program", "project", "provide", "quality", "range", "rate", "reason", "related", "report", 
    "research", "resource", "response", "result", "role", "section", "significant", "similar", 
    "situation", "source", "specific", "standard", "strategy", "structure", "study", "subject", 
    "support", "system", "term", "theory", "total", "type", "understand", "value", "variable", 
    "various",
    # Level 2 (High Frequency)
    "achieve", "acquire", "aspect", "assist", "assume", "authority", "challenge", "circumstance", 
    "comment", "conclude", "conduct", "consequence", "construct", "consumer", "contribution", 
    "create", "culture", "debate", "decline", "demonstrate", "distinction", "diversity", 
    "domestic", "emphasis", "enable", "energy", "ensure", "estimate", "evaluate", "expansion", 
    "exposure", "external", "facilitate", "fundamental", "generate", "grant", "hypothesis", 
    "illustration", "implication", "income", "indicate", "initial", "input", "instance", 
    "interaction", "internal", "investigate", "justify", "layer", "link", "locate", "mechanism", 
    "minor", "negative", "notion", "objective", "obtain", "occupy", "outcome", "parallel", 
    "participant", "perception", "perspective", "phase", "phenomenon", "philosophy", "physical", 
    "population", "predict", "previous", "primary", "principal", "principle", "prior", 
    "professional", "proportion", "psychology", "publish", "reaction", "registration", "reliance", 
    "remove", "require", "resolution", "respond", "sector", "secure", "select", "series", "shift", 
    "significance", "specify", "stability", "statistic", "status", "subsequent", "sufficient", 
    "sum", "summary", "survey", "task", "technical", "technique", "technology", "text", "tradition", 
    "transfer", "trend", "unique", "validity", "volume", "welfare"
}

# Consolidated list for backwards compatibility (optional)
LEXICAL_BUNDLES = SPOKEN_STANCE_BUNDLES + SPOKEN_DISCOURSE_BUNDLES + REFERENTIAL_BUNDLES

# Spoken Discourse Markers (Fluency & Coherence)
DISCOURSE_MARKERS = [
    "moving on to", "as I was saying", "to put it simply",
    "that reminds me", "by the way", "incidentally",
    "so to speak", "in other words", "consequently",
    "I mean", "you know", "of course", "in fact", "actually", "basically",
    "to be honest", "frankly speaking", "admittedly", "well", "so", "right", "okay",
    "mind you", "having said that", "on top of that", "funny enough", "believe it or not",
    "therefore", "however", "yet", "thus", "moreover", "furthermore", "nevertheless",
    # Added common spoken markers
    "oh", "definitely", "fortunately", "particularly", "personally", "absolutely",
    "suppose", "on the whole", "for instance", "simply", "kind of", "sort of"
]

# Filler Words (Appropriate use for coherence)
# Moved some to discourse markers if they serve a function
FILLERS = ["um", "uh", "er", "ah", "like", "sort of", "kind of"]

# Words that indicate strong intonation/attitude (Pronunciation/FC)
ATTITUDE_MARKERS = [
    "surprisingly", "unfortunately", "frankly", "honestly", "amazingly",
    "clearly", "obviously", "interestingly", "luckily", "admittedly",
    "undoubtedly", "seriously", "thankfully", "hopefully", "definitely", "absolutely",
    "undeniably", "arguably", "crucially", "fundamentally"
]

# Grammar Heuristics (Range & Accuracy)
COMPLEX_STRUCTURES = {
    "Conditional": [r"\bif\b", r"\bunless\b", r"\bprovided that\b", r"\bas long as\b"],
    "Relative Clause": [r"\bwho\b", r"\bwhich\b", r"\bthat\b", r"\bwhose\b", r"\bwhom\b", r"\bwhere\b", r"\bwhen\b"],
    "Subordinate Clause": [r"\balthough\b", r"\bthough\b", r"\bwhile\b", r"\bwhereas\b", r"\bbecause\b", r"\bsince\b", r"\bso that\b", r"\beven if\b"],
    "Passive Voice": [r"\b(is|are|was|were|been|being)\s+\w+(ed|en)\b"], 
    "Perfect Tense": [r"\b(have|has|had|'ve|'d)\s+\w+(ed|en)\b", r"\b(have|has|had|'ve|'d)\s+been\b"],
    "Modals": [r"\b(can|could|may|might|should|would|must|ought to|'ll|'d)\b"],
    "Continuous Aspect": [r"\b(am|is|are|was|were|'m|'s|'re)\s+\w+ing\b"],
    "Participle Clause": [r",\s\w+ing\b", r"^\w+ing\b"], # "..., giving it a vintage feel"
    "Cleft Sentences": [r"\bwhat\s(i|he|she|they|we)\s(like|hate|want|need)\s(is|was)\b", r"\bit\s(is|was)\s.*\sthat\b"],
    "Compound Sentence": [r",\s(but|and|so|or|yet|for|nor)\b"],
    "Tag Question": [r"(?:,|.)\s*(isn't|aren't|don't|doesn't|didn't|won't|can't|wouldn't|hasn't|haven't|shouldn't|is|are|do|does|did|will|can|would|has|have|should)\s*(it|he|she|they|we|you|i|there)\?"]
}

class TextInput(BaseModel):
    text: str
    duration: float = 0.0
    pronunciation_score: float = 0.0
    part: int = 1 # 1, 2, or 3 (Default to 1)
    user_id: str = "default"

# Context-Aware Vocab Mappings
# Structure: { simple_word: { context_keyword: [alternatives] } }
# "default" is used if no context matches
# Added metadata for ranking: difficulty (1-5), register (formal/neutral/spoken), topic
CONTEXT_VOCAB = {
    "good": {
        "default": [
            {"word": "solid", "diff": 2, "reg": "spoken"},
            {"word": "decent", "diff": 2, "reg": "spoken"},
            {"word": "positive", "diff": 2, "reg": "neutral"},
            {"word": "worthwhile", "diff": 3, "reg": "neutral"}
        ],
        "food": [
            {"word": "delicious", "diff": 2, "reg": "spoken"},
            {"word": "tasty", "diff": 1, "reg": "spoken"},
            {"word": "flavourful", "diff": 3, "reg": "neutral"},
            {"word": "scrumptious", "diff": 4, "reg": "spoken"}
        ],
        "person": [
            {"word": "supportive", "diff": 2, "reg": "spoken"},
            {"word": "reliable", "diff": 2, "reg": "neutral"},
            {"word": "considerate", "diff": 3, "reg": "neutral"},
            {"word": "dedicated", "diff": 3, "reg": "neutral"}
        ],
        "idea": [
            {"word": "sound", "diff": 2, "reg": "neutral"},
            {"word": "smart", "diff": 1, "reg": "spoken"},
            {"word": "innovative", "diff": 3, "reg": "neutral"}
        ],
        "health": [
            {"word": "healthy", "diff": 1, "reg": "spoken"},
            {"word": "fit", "diff": 1, "reg": "spoken"},
            {"word": "sound", "diff": 2, "reg": "neutral"}
        ],
        "performance": [
            {"word": "outstanding", "diff": 3, "reg": "neutral"},
            {"word": "brilliant", "diff": 2, "reg": "spoken"},
            {"word": "top-notch", "diff": 3, "reg": "spoken"}
        ],
        "weather": [
            {"word": "lovely", "diff": 1, "reg": "spoken"},
            {"word": "pleasant", "diff": 2, "reg": "neutral"},
            {"word": "gorgeous", "diff": 2, "reg": "spoken"}
        ]
    },
    "bad": {
        "default": [
            {"word": "rough", "diff": 2, "reg": "spoken"},
            {"word": "poor", "diff": 1, "reg": "neutral"},
            {"word": "not great", "diff": 1, "reg": "spoken"},
            {"word": "disappointing", "diff": 2, "reg": "neutral"}
        ],
        "weather": [
            {"word": "awful", "diff": 1, "reg": "spoken"},
            {"word": "terrible", "diff": 1, "reg": "spoken"},
            {"word": "harsh", "diff": 2, "reg": "neutral"}
        ],
        "situation": [
            {"word": "tricky", "diff": 2, "reg": "spoken"},
            {"word": "tough", "diff": 1, "reg": "spoken"},
            {"word": "messy", "diff": 2, "reg": "spoken"}
        ],
        "behavior": [
            {"word": "rude", "diff": 1, "reg": "spoken"},
            {"word": "inappropriate", "diff": 3, "reg": "neutral"},
            {"word": "unacceptable", "diff": 2, "reg": "neutral"}
        ],
        "health": [
            {"word": "poor", "diff": 1, "reg": "neutral"},
            {"word": "getting worse", "diff": 1, "reg": "spoken"},
            {"word": "deteriorating", "diff": 4, "reg": "formal"}
        ],
        "quality": [
            {"word": "rubbish", "diff": 2, "reg": "spoken"},
            {"word": "shoddy", "diff": 3, "reg": "spoken"},
            {"word": "inferior", "diff": 3, "reg": "neutral"}
        ]
    },
    "happy": {
        "default": [
            {"word": "pleased", "diff": 2, "reg": "neutral"},
            {"word": "glad", "diff": 1, "reg": "spoken"},
            {"word": "content", "diff": 2, "reg": "neutral"},
            {"word": "chuffed", "diff": 3, "reg": "spoken"}
        ],
        "extremely": [
            {"word": "thrilled", "diff": 3, "reg": "spoken"},
            {"word": "over the moon", "diff": 3, "reg": "spoken"},
            {"word": "ecstatic", "diff": 4, "reg": "spoken"}
        ],
        "peaceful": [
            {"word": "relaxed", "diff": 1, "reg": "spoken"},
            {"word": "at ease", "diff": 2, "reg": "spoken"},
            {"word": "calm", "diff": 1, "reg": "neutral"}
        ],
        "achievement": [
            {"word": "proud", "diff": 1, "reg": "spoken"},
            {"word": "fulfilled", "diff": 3, "reg": "neutral"},
            {"word": "buzzing", "diff": 3, "reg": "spoken"}
        ]
    },
    "important": {
        "default": [
            {"word": "key", "diff": 1, "reg": "spoken"},
            {"word": "crucial", "diff": 3, "reg": "neutral"},
            {"word": "meaningful", "diff": 2, "reg": "neutral"},
            {"word": "a big deal", "diff": 2, "reg": "spoken"}
        ],
        "necessary": [
            {"word": "essential", "diff": 3, "reg": "neutral"},
            {"word": "crucial", "diff": 3, "reg": "neutral"},
            {"word": "vital", "diff": 3, "reg": "neutral"}
        ],
        "center": [
            {"word": "central", "diff": 2, "reg": "neutral"},
            {"word": "core", "diff": 2, "reg": "spoken"},
            {"word": "pivotal", "diff": 4, "reg": "formal"}
        ],
        "future": [
            {"word": "life-changing", "diff": 2, "reg": "spoken"},
            {"word": "far-reaching", "diff": 4, "reg": "formal"},
            {"word": "game-changing", "diff": 3, "reg": "spoken"}
        ]
    },
    "think": {
        "default": [
            {"word": "feel", "diff": 1, "reg": "spoken"},
            {"word": "believe", "diff": 1, "reg": "neutral"},
            {"word": "reckon", "diff": 3, "reg": "spoken"},
            {"word": "figure", "diff": 2, "reg": "spoken"}
        ],
        "opinion": [
            {"word": "reckon", "diff": 3, "reg": "spoken"},
            {"word": "suppose", "diff": 2, "reg": "spoken"},
            {"word": "would say", "diff": 2, "reg": "spoken"}
        ],
        "deep": [
            {"word": "reflect on", "diff": 3, "reg": "neutral"},
            {"word": "mull over", "diff": 3, "reg": "spoken"},
            {"word": "contemplate", "diff": 4, "reg": "formal"}
        ],
        "guess": [
            {"word": "guess", "diff": 1, "reg": "spoken"},
            {"word": "suspect", "diff": 3, "reg": "neutral"},
            {"word": "speculate", "diff": 4, "reg": "formal"}
        ]
    },
    "big": {
        "default": [
            {"word": "huge", "diff": 1, "reg": "spoken"},
            {"word": "massive", "diff": 2, "reg": "spoken"},
            {"word": "major", "diff": 2, "reg": "neutral"},
            {"word": "enormous", "diff": 3, "reg": "neutral"}
        ],
        "size": [
            {"word": "massive", "diff": 2, "reg": "spoken"},
            {"word": "gigantic", "diff": 3, "reg": "spoken"},
            {"word": "enormous", "diff": 3, "reg": "neutral"}
        ],
        "scope": [
            {"word": "wide-ranging", "diff": 3, "reg": "neutral"},
            {"word": "broad", "diff": 2, "reg": "neutral"},
            {"word": "extensive", "diff": 3, "reg": "formal"}
        ],
        "impact": [
            {"word": "huge", "diff": 1, "reg": "spoken"},
            {"word": "major", "diff": 2, "reg": "neutral"},
            {"word": "significant", "diff": 3, "reg": "formal"}
        ]
    },
    "like": {
        "default": [
            {"word": "enjoy", "diff": 1, "reg": "neutral"},
            {"word": "be into", "diff": 2, "reg": "spoken"},
            {"word": "be fond of", "diff": 3, "reg": "neutral"},
            {"word": "prefer", "diff": 2, "reg": "neutral"}
        ],
        "love": [
            {"word": "adore", "diff": 3, "reg": "spoken"},
            {"word": "be crazy about", "diff": 2, "reg": "spoken"},
            {"word": "be passionate about", "diff": 3, "reg": "neutral"}
        ],
        "admire": [
            {"word": "look up to", "diff": 2, "reg": "spoken"},
            {"word": "respect", "diff": 2, "reg": "neutral"},
            {"word": "admire", "diff": 3, "reg": "neutral"}
        ],
        "food": [
            {"word": "love", "diff": 1, "reg": "spoken"},
            {"word": "be a fan of", "diff": 2, "reg": "spoken"},
            {"word": "have a taste for", "diff": 3, "reg": "spoken"}
        ]
    },
    "help": {
        "default": [
            {"word": "give a hand", "diff": 2, "reg": "spoken"},
            {"word": "support", "diff": 2, "reg": "neutral"},
            {"word": "pitch in", "diff": 3, "reg": "spoken"},
            {"word": "back up", "diff": 2, "reg": "spoken"}
        ],
        "problem": [
            {"word": "sort out", "diff": 2, "reg": "spoken"},
            {"word": "deal with", "diff": 2, "reg": "spoken"},
            {"word": "fix", "diff": 1, "reg": "spoken"}
        ],
        "person": [
            {"word": "lend a hand", "diff": 2, "reg": "spoken"},
            {"word": "guide", "diff": 2, "reg": "neutral"},
            {"word": "mentor", "diff": 3, "reg": "neutral"}
        ],
        "process": [
            {"word": "speed up", "diff": 2, "reg": "spoken"},
            {"word": "make easier", "diff": 1, "reg": "spoken"},
            {"word": "streamline", "diff": 4, "reg": "formal"}
        ]
    },
    "problem": {
        "default": [
            {"word": "issue", "diff": 2, "reg": "neutral"},
            {"word": "challenge", "diff": 2, "reg": "neutral"},
            {"word": "hassle", "diff": 2, "reg": "spoken"},
            {"word": "headache", "diff": 2, "reg": "spoken"}
        ],
        "serious": [
            {"word": "crisis", "diff": 3, "reg": "neutral"},
            {"word": "nightmare", "diff": 2, "reg": "spoken"},
            {"word": "dilemma", "diff": 3, "reg": "neutral"}
        ],
        "obstacle": [
            {"word": "hurdle", "diff": 3, "reg": "spoken"},
            {"word": "stumbling block", "diff": 3, "reg": "spoken"},
            {"word": "barrier", "diff": 3, "reg": "neutral"}
        ],
        "complex": [
            {"word": "tricky situation", "diff": 2, "reg": "spoken"},
            {"word": "complication", "diff": 3, "reg": "neutral"},
            {"word": "mess", "diff": 1, "reg": "spoken"}
        ]
    },
    "interesting": {
        "default": [
            {"word": "fascinating", "diff": 3, "reg": "neutral"},
            {"word": "cool", "diff": 1, "reg": "spoken"},
            {"word": "engaging", "diff": 3, "reg": "neutral"},
            {"word": "eye-opening", "diff": 3, "reg": "spoken"}
        ],
        "book": [
            {"word": "gripping", "diff": 4, "reg": "spoken"},
            {"word": "a real page-turner", "diff": 3, "reg": "spoken"},
            {"word": "unputdownable", "diff": 4, "reg": "spoken"}
        ],
        "person": [
            {"word": "fascinating", "diff": 3, "reg": "neutral"},
            {"word": "charismatic", "diff": 4, "reg": "neutral"},
            {"word": "one of a kind", "diff": 2, "reg": "spoken"}
        ],
        "idea": [
            {"word": "thought-provoking", "diff": 4, "reg": "neutral"},
            {"word": "intriguing", "diff": 4, "reg": "neutral"},
            {"word": "mind-blowing", "diff": 3, "reg": "spoken"}
        ]
    },
    "get": {
        "default": [
            {"word": "pick up", "diff": 2, "reg": "spoken"},
            {"word": "grab", "diff": 1, "reg": "spoken"},
            {"word": "receive", "diff": 2, "reg": "neutral"},
            {"word": "land", "diff": 2, "reg": "spoken"}
        ],
        "understand": [
            {"word": "grasp", "diff": 3, "reg": "spoken"},
            {"word": "figure out", "diff": 2, "reg": "spoken"},
            {"word": "wrap my head around", "diff": 3, "reg": "spoken"}
        ],
        "become": [
            {"word": "turn", "diff": 1, "reg": "neutral"},
            {"word": "grow", "diff": 1, "reg": "neutral"},
            {"word": "end up", "diff": 2, "reg": "spoken"}
        ],
        "arrive": [
            {"word": "reach", "diff": 2, "reg": "neutral"},
            {"word": "make it to", "diff": 2, "reg": "spoken"},
            {"word": "show up at", "diff": 2, "reg": "spoken"}
        ]
    },
    "do": {
        "default": [
            {"word": "handle", "diff": 2, "reg": "spoken"},
            {"word": "take care of", "diff": 2, "reg": "spoken"},
            {"word": "get done", "diff": 1, "reg": "spoken"},
            {"word": "work on", "diff": 1, "reg": "spoken"}
        ],
        "activity": [
            {"word": "get into", "diff": 2, "reg": "spoken"},
            {"word": "take part in", "diff": 2, "reg": "neutral"},
            {"word": "try out", "diff": 2, "reg": "spoken"}
        ],
        "job": [
            {"word": "finish", "diff": 1, "reg": "spoken"},
            {"word": "complete", "diff": 2, "reg": "neutral"},
            {"word": "wrap up", "diff": 2, "reg": "spoken"}
        ],
        "research": [
            {"word": "look into", "diff": 2, "reg": "spoken"},
            {"word": "explore", "diff": 3, "reg": "neutral"},
            {"word": "dig into", "diff": 3, "reg": "spoken"}
        ]
    },
    "nice": {
        "default": [
            {"word": "lovely", "diff": 1, "reg": "spoken"},
            {"word": "pleasant", "diff": 2, "reg": "neutral"},
            {"word": "great", "diff": 1, "reg": "spoken"},
            {"word": "enjoyable", "diff": 2, "reg": "neutral"}
        ],
        "person": [
            {"word": "lovely", "diff": 1, "reg": "spoken"},
            {"word": "friendly", "diff": 1, "reg": "spoken"},
            {"word": "warm", "diff": 2, "reg": "spoken"},
            {"word": "kind-hearted", "diff": 3, "reg": "spoken"}
        ],
        "weather": [
            {"word": "gorgeous", "diff": 2, "reg": "spoken"},
            {"word": "lovely", "diff": 1, "reg": "spoken"},
            {"word": "mild", "diff": 3, "reg": "neutral"}
        ],
        "appearance": [
            {"word": "gorgeous", "diff": 2, "reg": "spoken"},
            {"word": "attractive", "diff": 2, "reg": "neutral"},
            {"word": "good-looking", "diff": 2, "reg": "spoken"}
        ]
    },
    "lot": {
        "default": [
            {"word": "loads", "diff": 2, "reg": "spoken"},
            {"word": "tons", "diff": 2, "reg": "spoken"},
            {"word": "quite a few", "diff": 2, "reg": "spoken"},
            {"word": "plenty", "diff": 2, "reg": "neutral"}
        ],
        "quantity": [
            {"word": "loads of", "diff": 2, "reg": "spoken"},
            {"word": "heaps of", "diff": 2, "reg": "spoken"},
            {"word": "tons of", "diff": 2, "reg": "spoken"}
        ],
        "degree": [
            {"word": "a great deal", "diff": 3, "reg": "neutral"},
            {"word": "quite a bit", "diff": 2, "reg": "spoken"},
            {"word": "way more", "diff": 2, "reg": "spoken"}
        ]
    },
    "say": {
        "default": [
            {"word": "mention", "diff": 2, "reg": "neutral"},
            {"word": "point out", "diff": 2, "reg": "spoken"},
            {"word": "go on about", "diff": 2, "reg": "spoken"},
            {"word": "bring up", "diff": 2, "reg": "spoken"}
        ],
        "opinion": [
            {"word": "argue", "diff": 3, "reg": "neutral"},
            {"word": "claim", "diff": 2, "reg": "neutral"},
            {"word": "reckon", "diff": 3, "reg": "spoken"}
        ],
        "reply": [
            {"word": "answer", "diff": 1, "reg": "neutral"},
            {"word": "come back with", "diff": 2, "reg": "spoken"},
            {"word": "respond", "diff": 3, "reg": "neutral"}
        ],
        "suggest": [
            {"word": "hint at", "diff": 3, "reg": "spoken"},
            {"word": "imply", "diff": 4, "reg": "neutral"},
            {"word": "suggest", "diff": 2, "reg": "neutral"}
        ]
    },
    "really": {
        "default": [
            {"word": "genuinely", "diff": 3, "reg": "neutral"},
            {"word": "truly", "diff": 2, "reg": "neutral"},
            {"word": "honestly", "diff": 2, "reg": "spoken"},
            {"word": "seriously", "diff": 2, "reg": "spoken"}
        ],
        "very": [
            {"word": "incredibly", "diff": 3, "reg": "spoken"},
            {"word": "extremely", "diff": 3, "reg": "neutral"},
            {"word": "absolutely", "diff": 2, "reg": "spoken"}
        ],
        "fact": [
            {"word": "honestly", "diff": 2, "reg": "spoken"},
            {"word": "actually", "diff": 1, "reg": "spoken"},
            {"word": "in fact", "diff": 2, "reg": "neutral"}
        ]
    }
}

USER_VOCAB_HISTORY = {}

class FeedbackRequest(BaseModel):
    original: str
    recommended: str
    action: str
    user_id: str = "default"

_native_collocations = None
_native_collocation_set = None
_native_adj_by_noun = None
_native_adv_by_adj = None

def _get_native_collocations():
    global _native_collocations, _native_collocation_set, _native_adj_by_noun, _native_adv_by_adj
    if _native_collocations is not None:
        return _native_collocations, _native_collocation_set, _native_adj_by_noun, _native_adv_by_adj
    path = os.path.join(os.path.dirname(__file__), "native_collocations.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
    collocations = []
    colloc_set = set()
    adj_by_noun = {}
    adv_by_adj = {}
    for item in data:
        if not isinstance(item, str):
            continue
        phrase = item.strip()
        if not phrase:
            continue
        phrase_l = phrase.lower()
        collocations.append(phrase)
        colloc_set.add(phrase_l)
        parts = phrase_l.split()
        if len(parts) == 2:
            a, b = parts
            if b not in adj_by_noun:
                adj_by_noun[b] = set()
            adj_by_noun[b].add(a)
            if b not in adv_by_adj:
                adv_by_adj[b] = set()
            adv_by_adj[b].add(a)
        elif len(parts) >= 2:
            head = parts[-1]
            mod = parts[-2]
            if head not in adj_by_noun:
                adj_by_noun[head] = set()
            adj_by_noun[head].add(mod)
    _native_collocations = collocations
    _native_collocation_set = colloc_set
    _native_adj_by_noun = adj_by_noun
    _native_adv_by_adj = adv_by_adj
    return _native_collocations, _native_collocation_set, _native_adj_by_noun, _native_adv_by_adj


# Context Keywords for detection
CONTEXT_KEYWORDS = {
    "food": ["eat", "taste", "delicious", "meal", "restaurant", "dish", "flavor", "cooking", "cuisine", "lunch", "dinner", "breakfast"],
    "person": ["man", "woman", "friend", "teacher", "parent", "people", "guy", "girl", "personality", "character", "neighbor", "colleague"],
    "idea": ["plan", "thought", "concept", "theory", "suggestion", "proposal", "strategy", "opinion", "belief", "view"],
    "health": ["sick", "ill", "doctor", "medicine", "hospital", "pain", "body", "exercise", "diet", "fitness", "wellbeing"],
    "weather": ["rain", "sun", "cloud", "storm", "wind", "snow", "temperature", "climate", "forecast", "hot", "cold"],
    "performance": ["job", "work", "exam", "test", "result", "score", "play", "music", "movie", "acting", "presentation"],
    "situation": ["problem", "issue", "event", "circumstance", "case", "scenario", "condition", "state", "situation"],
    "behavior": ["act", "rude", "polite", "kind", "mean", "attitude", "conduct", "manner", "behave"],
    "achievement": ["win", "success", "goal", "award", "prize", "accomplish", "finish", "complete", "succeed"],
    "necessary": ["need", "must", "require", "have to", "essential", "vital", "important"],
    "future": ["will", "going to", "plan", "dream", "goal", "career", "life", "tomorrow", "year", "aspirations"],
    "deep": ["why", "reason", "philosophy", "meaning", "understand", "learn", "study", "analyze", "reflect"],
    "scope": ["project", "area", "field", "range", "cover", "include", "involve", "extent"],
    "impact": ["change", "affect", "effect", "result", "consequence", "outcome", "influence"],
    "quality": ["product", "item", "material", "make", "brand", "standard"],
    "love": ["heart", "romance", "passion", "family", "marriage", "relationship"],
    "admire": ["hero", "idol", "leader", "role model", "inspiration"],
    "process": ["system", "method", "way", "procedure", "workflow", "operation"],
    "serious": ["danger", "threat", "emergency", "urgent", "critical", "severe"],
    "obstacle": ["block", "stop", "prevent", "hinder", "limit", "restrict"],
    "complex": ["hard", "difficult", "confusing", "complicated", "tricky"],
    "book": ["read", "story", "novel", "author", "plot", "character", "literature"]
}

_context_vectors = None

def _get_context_vectors():
    global _context_vectors
    if _context_vectors is not None:
        return _context_vectors
    nlp = get_spacy()
    vectors = {}
    for context, keywords in CONTEXT_KEYWORDS.items():
        valid_vecs = []
        for k in keywords:
            v = nlp(k).vector
            if np.any(v):
                valid_vecs.append(v)
        if valid_vecs:
            vec = np.mean(valid_vecs, axis=0)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors[context] = vec
    _context_vectors = vectors
    return _context_vectors

def detect_context(text: str) -> List[str]:
    detected = []
    text_lower = text.lower()
    for context, keywords in CONTEXT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(context)
    nlp = get_spacy()
    doc = nlp(text)
    if doc.vector_norm:
        doc_vec = doc.vector / doc.vector_norm
        for context, vec in _get_context_vectors().items():
            sim = float(np.dot(doc_vec, vec))
            if sim > 0.55:
                detected.append(context)
    return list(set(detected))

def get_context_aware_word_recommendations(
    doc,
    part: int,
    user_profile,
    scoring_system,
    min_diff: float,
    max_diff: float,
    used_words: set,
    history_allows_fn,
    record_shown_fn,
    max_alternatives: int = 5,
):
    if not doc.vector_norm:
        return None

    nlp = get_spacy()
    text_lower = doc.text.lower()
    keyword_contexts = set()
    for ctx, keywords in CONTEXT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                keyword_contexts.add(ctx)
                break

    contexts = set(keyword_contexts)
    doc_vec = doc.vector / doc.vector_norm
    top_vec_contexts = []
    if not contexts:
        ctx_vecs = _get_context_vectors()
        sims = []
        for ctx, vec in ctx_vecs.items():
            sims.append((float(np.dot(doc_vec, vec)), ctx))
        sims.sort(reverse=True)
        top_vec_contexts = [ctx for sim, ctx in sims[:2] if sim > 0.7]
        contexts.update(top_vec_contexts)

    if not contexts:
        return None

    if len(contexts) > 3:
        if keyword_contexts:
            contexts = set(list(keyword_contexts)[:3])
        else:
            contexts = set(top_vec_contexts[:2])

    candidates = []
    for _, options in CONTEXT_VOCAB.items():
        for ctx in contexts:
            if ctx not in options:
                continue
            for cand in options.get(ctx, []):
                cand_copy = cand.copy()
                cand_copy["source"] = "context"
                cand_copy["context"] = ctx
                candidates.append(cand_copy)

    if not candidates:
        return None

    filtered = []
    for cand in candidates:
        cand_word_raw = str(cand.get("word", "")).strip()
        cand_key = cand_word_raw.lower()
        if not cand_key:
            continue
        if cand_key in used_words:
            continue
        reg = str(cand.get("reg", "neutral")).lower()
        if part in {1, 2} and reg in {"academic", "literary", "archaic"}:
            continue
        if part == 3 and reg in {"archaic", "literary"}:
            continue
        word_diff = cand.get("diff", 3)
        if word_diff < min_diff - 1 or word_diff > max_diff + 1:
            continue
        if not history_allows_fn(cand_word_raw):
            continue

        cdoc = nlp(cand_word_raw)
        if cdoc.vector_norm:
            cand["context_similarity"] = float(np.dot(doc_vec, cdoc.vector / cdoc.vector_norm))
        else:
            cand["context_similarity"] = 0.0

        filtered.append(cand)


    if not filtered:
        return None

    ranked = scoring_system.rank_candidates(filtered, doc.text)
    if not ranked:
        return None

    best = []
    seen = set()
    for _, cand in ranked:
        w = str(cand.get("word", "")).strip()
        k = w.lower()
        if not k or k in seen:
            continue
        best.append(w)
        seen.add(k)
        if len(best) >= max_alternatives:
            break

    if not best:
        return None

    for w in best:
        record_shown_fn(w)

    ctx_label = ", ".join(sorted(set(contexts)))
    return {
        "original": "Topic vocabulary",
        "alternatives": best,
        "reason": f"Context-aware suggestions for: {ctx_label}",
        "score": float(ranked[0][0]),
        "difficulty_range": f"{min_diff:.1f}-{max_diff:.1f}",
        "user_level": user_profile.proficiency_level,
    }


# ============================================================================
# ENHANCED WORD RECOMMENDATION FUNCTION
# Following the AI-Driven Pedagogical Workflow (Figure 4)
# ============================================================================

def _norm_rec_original(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value).strip()
    return ""


def _norm_rec_alternatives(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for alt in value:
        if isinstance(alt, str):
            alt_text = alt.strip()
            if alt_text:
                cleaned.append(alt_text)
    return cleaned


def _norm_rec_reason(value: Any) -> str:
    if isinstance(value, str):
        txt = value.strip()
        if txt:
            return txt
    return "Contextual improvement"


def _safe_add_str_to_set(s: set, value: Any) -> bool:
    """Add value to set only if it is a hashable string. Returns True if added."""
    if isinstance(value, str):
        s.add(value.lower())
        return True
    return False


def _build_existing_originals_set(items: List[Dict[str, Any]]) -> set:
    dedup = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        orig = _norm_rec_original(item.get("original"))
        if orig:
            try:
                dedup.add(orig.lower())
            except TypeError:
                pass  # Skip unhashable values
    return dedup


def smoke_test_llm_merge_normalization() -> Dict[str, Any]:
    """
    Minimal smoke test for recommendation normalization.
    Safe to run manually in a Python shell:
      from main import smoke_test_llm_merge_normalization
      print(smoke_test_llm_merge_normalization())
    """
    recommendations: List[Dict[str, Any]] = [
        {"original": "good", "alternatives": ["great"], "reason": "base"},
        {"original": {"bad": "shape"}, "alternatives": [], "reason": "invalid original"},
        {"original": 123, "alternatives": ["numeric original allowed"], "reason": None},
    ]
    existing = _build_existing_originals_set(recommendations)

    llm_payload = [
        {"original": "good", "alternatives": ["excellent"], "reason": "dup"},
        {"original": {"oops": 1}, "alternatives": ["x"], "reason": "bad"},
        {"original": "nice", "alternatives": [" pleasant ", {"wrong": True}, "friendly"], "reason": {"why": "bad type"}},
    ]

    merged_count = 0
    for rec in llm_payload:
        if not isinstance(rec, dict):
            continue
        orig = _norm_rec_original(rec.get("original"))
        if not orig or orig.lower() in existing:
            continue
        recommendations.append({
            "original": orig,
            "alternatives": _norm_rec_alternatives(rec.get("alternatives")),
            "reason": _norm_rec_reason(rec.get("reason")) + " (AI-Recommended)",
        })
        existing.add(orig.lower())
        merged_count += 1

    return {
        "ok": True,
        "existing_size": len(existing),
        "merged_count": merged_count,
        "last_item": recommendations[-1] if recommendations else None,
    }


def init_ontology_trajectory_system() -> None:
    """Initialize ontology/persistence objects once."""
    global STATE_STORE, VOCAB_ONTOLOGY, TRAJECTORY_PLANNER
    if not ONTOLOGY_TRAJECTORY_ENABLED:
        return
    if STATE_STORE is None and SQLitePersistence is not None:
        try:
            STATE_STORE = SQLitePersistence(TRAJECTORY_DB_PATH)
            logger.info(f"Ontology state store ready at {TRAJECTORY_DB_PATH}")
        except Exception as e:
            logger.warning(f"Failed to initialize trajectory sqlite store: {e}")
    if VOCAB_ONTOLOGY is None and VocabularyOntology is not None:
        try:
            seed_path = os.path.join(os.path.dirname(__file__), "data", "ontology_seed.json")
            VOCAB_ONTOLOGY = VocabularyOntology(CONTEXT_VOCAB, CONTEXT_KEYWORDS, seed_path=seed_path)
            logger.info("Vocabulary ontology initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize vocabulary ontology: {e}")
            VOCAB_ONTOLOGY = None
    if TRAJECTORY_PLANNER is None and TrajectoryPlanner is not None and VOCAB_ONTOLOGY is not None:
        try:
            TRAJECTORY_PLANNER = TrajectoryPlanner(VOCAB_ONTOLOGY)
            logger.info("Trajectory planner initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize trajectory planner: {e}")
            TRAJECTORY_PLANNER = None


def get_learner_state(user_id: str = "default"):
    if LearnerState is None:
        return None
    user_id = (user_id or "default").strip() or "default"
    state = LEARNER_STATES.get(user_id)
    if state is not None:
        return state
    if ONTOLOGY_TRAJECTORY_ENABLED and STATE_STORE is not None:
        try:
            state = STATE_STORE.load_state(user_id)
        except Exception:
            state = LearnerState(user_id=user_id)
    else:
        state = LearnerState(user_id=user_id)
    LEARNER_STATES[user_id] = state
    return state


def persist_learner_state(user_id: str = "default") -> None:
    if not ONTOLOGY_TRAJECTORY_ENABLED or STATE_STORE is None:
        return
    state = LEARNER_STATES.get(user_id)
    if state is None:
        return
    try:
        STATE_STORE.save_state(user_id, state)
    except Exception as e:
        logger.warning(f"Failed to save learner state for {user_id}: {e}")


def _extract_weak_areas_from_breakdown(breakdown: Dict[str, Any]) -> List[str]:
    weak = []
    if not isinstance(breakdown, dict):
        return weak
    for key in ("fluency", "lexical", "grammar", "pronunciation"):
        try:
            value = float(breakdown.get(key, 0))
        except Exception:
            value = 0.0
        if value and value < 6.5:
            weak.append(key)
    return weak


def update_session_remap(
    user_id: str,
    text: str,
    part: int,
    breakdown: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    accepted_words: Optional[List[str]] = None,
) -> None:
    """Session remapper: updates multidimensional learner state and concept mastery."""
    state = get_learner_state(user_id)
    if state is None:
        return
    accepted_words = accepted_words or []
    weak_areas = _extract_weak_areas_from_breakdown(breakdown)

    # Update criterion dimensions with bounded learning rate.
    alpha = 0.15
    for dim in ("fluency", "lexical", "grammar", "pronunciation"):
        if dim not in breakdown:
            continue
        try:
            observed = float(breakdown.get(dim, 5.0))
        except Exception:
            continue
        prev = float(state.domain_bands.get(dim, 5.0))
        updated = (1 - alpha) * prev + alpha * observed
        state.domain_bands[dim] = max(1.0, min(9.0, updated))

    state.fluency = float(state.domain_bands.get("fluency", state.fluency))
    state.grammar_complexity = float(state.domain_bands.get("grammar", state.grammar_complexity))
    state.register_score = max(1.0, min(9.0, float(state.domain_bands.get("lexical", 5.0))))
    state.discourse_score = max(1.0, min(9.0, float(state.domain_bands.get("fluency", 5.0))))

    # Concept mapping from recommendations and accepted words.
    touched_concepts = set(accepted_words)
    for rec in recommendations or []:
        if not isinstance(rec, dict):
            continue
        orig = _norm_rec_original(rec.get("original"))
        if orig:
            touched_concepts.add(orig)
        for alt in _norm_rec_alternatives(rec.get("alternatives")):
            touched_concepts.add(alt.lower())

    for concept in touched_concepts:
        prev = float(state.concept_mastery.get(concept, 0.0))
        delta = 0.08
        if concept in accepted_words:
            delta += 0.12
        if weak_areas:
            delta -= 0.03
        state.concept_mastery[concept] = max(0.0, min(1.0, prev + delta))

    # Lightweight replanning trigger: move step if current target is already mastered.
    if state.trajectory_plan and 0 <= state.trajectory_step < len(state.trajectory_plan):
        current = state.trajectory_plan[state.trajectory_step]
        current_concept = str(current.get("concept", "")).strip().lower()
        if current_concept and float(state.concept_mastery.get(current_concept, 0.0)) >= 0.75:
            state.trajectory_step = min(state.trajectory_step + 1, max(0, len(state.trajectory_plan) - 1))

    persist_learner_state(user_id)
    if ONTOLOGY_TRAJECTORY_ENABLED and STATE_STORE is not None:
        try:
            STATE_STORE.append_session_log(
                user_id,
                {
                    "part": part,
                    "weak_areas": weak_areas,
                    "text_len": len(text or ""),
                    "recommendation_count": len(recommendations or []),
                    "accepted_words": accepted_words,
                    "overall_band": state.overall_band(),
                },
            )
        except Exception:
            pass

def get_smart_recommendations(text: str, part: int = 1, user_id: str = "default") -> List[Dict]:
    """
    AI-Driven Pedagogical Word Recommendation System.
    
    Workflow (from IEEE ICMLAS 2025 paper):
    1. User Profile Creation - Load user skills, objectives, preferences
    2. AI-Engine Analyses User Data - NLP analysis of input text
    3. Personalized Word Recommendation - Context-aware, difficulty-appropriate
    4. Interactive Exercises & Feedback - (handled by feedback endpoint)
    5. Performance Logged in Learning Repository - Track all interactions
    6. AI Model Refinement - Adaptive difficulty adjustment
    7. Progress Monitoring Dashboard - (handled by progress endpoint)
    
    Args:
        text: User's spoken/written text to analyze
        part: IELTS speaking part (1, 2, or 3)
        user_id: User identifier for personalization
        
    Returns:
        List of word recommendations with alternatives and reasons
    """
    user_id = (user_id or "default").strip() or "default"
    recommendations = []
    trajectory_context: Dict[str, Any] = {}
    
    # Step 1: Load User Profile (User Profile Creation)
    user_profile = get_user_profile(user_id)
    learning_repo = get_learning_repo(user_id)
    
    # Update session tracking
    user_profile.total_sessions += 1
    user_profile.last_active = datetime.now()
    
    # Initialize AI systems
    scoring_system = AIWordScoringSystem(user_profile, learning_repo)
    adaptive_engine = AdaptiveLearningEngine(user_profile, learning_repo)
    
    # Get recommended difficulty range based on user's current level
    min_diff, optimal_diff, max_diff = adaptive_engine.get_recommended_difficulty_range()
    if ONTOLOGY_TRAJECTORY_ENABLED:
        init_ontology_trajectory_system()
    
    # Step 2: AI-Engine Analyses User Data (NLP Processing)
    nlp = get_spacy()
    doc = nlp(text)
    text_lower = text.lower()
    _, colloc_set, adj_by_noun, adv_by_adj = _get_native_collocations()

    TARGET_POS_MAP = {
        "good": "JJ", "bad": "JJ", "big": "JJ", "important": "JJ", "interesting": "JJ",
        "happy": "JJ", "serious": "JJ", "complex": "JJ", "necessary": "JJ",
        "think": "VB", "like": "VB", "help": "VB", "get": "VB", "do": "VB", "say": "VB",
        "nice": "JJ", "lot": "NN", "really": "RB"
    }
    
    seen_words = set()
    used_words = {t.text.lower() for t in doc if not t.is_space}
    now = time.time()
    cooldown_sec = 120.0
    seen_originals = set()
    trajectory_targets = set()
    learner_state = get_learner_state(user_id) if ONTOLOGY_TRAJECTORY_ENABLED else None
    if ONTOLOGY_TRAJECTORY_ENABLED and learner_state is not None and VOCAB_ONTOLOGY is not None and TRAJECTORY_PLANNER is not None:
        detected_concepts = VOCAB_ONTOLOGY.map_text_to_concepts(text, part=part)
        weak_areas = []
        session_signal = {
            "detected_concepts": detected_concepts,
            "weak_areas": weak_areas,
            "register_hint": user_profile.register_preference or "neutral",
        }
        steps = TRAJECTORY_PLANNER.plan_trajectory(
            state=learner_state,
            session_signal=session_signal,
            target_band=user_profile.target_level,
            min_diff=min_diff,
            max_diff=max_diff,
        )
        learner_state.trajectory_plan = [s.to_dict() for s in steps]
        if learner_state.trajectory_step >= len(learner_state.trajectory_plan):
            learner_state.trajectory_step = 0
        if learner_state.trajectory_plan:
            current_step = learner_state.trajectory_plan[learner_state.trajectory_step]
            trajectory_context = {
                "trajectory_step": current_step,
                "target_concepts": [s["concept"] for s in learner_state.trajectory_plan[:3]],
                "register_goal": user_profile.register_preference or "neutral",
            }
            current_concept = str(current_step.get("concept", "")).strip().lower()
            if current_concept:
                for n in VOCAB_ONTOLOGY.get_candidate_neighbors(
                    [current_concept],
                    max_hops=1,
                    register_hint=user_profile.register_preference or "neutral",
                ):
                    trajectory_targets.add(str(n).lower())
                for pre in VOCAB_ONTOLOGY.get_prerequisite_chain(current_concept):
                    trajectory_targets.add(str(pre).lower())
            for step in learner_state.trajectory_plan[:4]:
                concept = str(step.get("concept", "")).strip().lower()
                if concept:
                    trajectory_targets.add(concept)
        persist_learner_state(user_id)
        if STATE_STORE is not None:
            try:
                STATE_STORE.save_trajectory(user_id, learner_state.trajectory_plan)
            except Exception:
                pass
    
    # Track words user has used (for vocabulary growth monitoring)
    for token in doc:
        word_lower = token.text.lower()
        if word_lower in user_profile.vocabulary_introduced:
            # User has used a previously introduced word!
            learning_repo.log_interaction(word_lower, "used", text[:50])
            user_profile.vocabulary_learning.add(word_lower)

    def _history_allows(candidate_text: str) -> bool:
        """Check if candidate should be shown based on learning repository."""
        key = candidate_text.strip().lower()
        if not key:
            return False
        
        # Check learning repository for engagement metrics
        stats = learning_repo.get_word_stats(key)
        
        # Skip if user has ignored this word
        if stats.get("ignored_count", 0) > 0:
            return False
            
        # Skip if word is already mastered
        if stats.get("mastery_level", 0) >= 4:
            return False
            
        # Check cooldown period
        last_interaction = stats.get("last_interaction")
        if last_interaction:
            try:
                last_dt = datetime.fromisoformat(last_interaction)
                seconds_since = (datetime.now() - last_dt).total_seconds()
                if seconds_since < cooldown_sec:
                    return False
            except:
                pass
                
        # Legacy check for backwards compatibility
        h = USER_VOCAB_HISTORY.get(key, {})
        if h.get("ignored_count", 0) > 0:
            return False
        last_shown = h.get("last_shown", 0.0)
        if last_shown and (now - float(last_shown)) < cooldown_sec:
            return False
            
        return True

    def _record_shown(candidate_text: str):
        """Record that a word recommendation was shown to user."""
        key = candidate_text.strip().lower()
        if not key:
            return
            
        # Log to learning repository (Step 5: Performance Logged)
        learning_repo.log_interaction(key, "shown", text[:50])
        
        # Track introduced vocabulary
        user_profile.vocabulary_introduced.add(key)
        
        # Legacy tracking for backwards compatibility
        h = USER_VOCAB_HISTORY.get(key)
        if not h:
            h = {"shown_count": 0, "accepted_count": 0, "ignored_count": 0, "last_shown": 0.0}
            USER_VOCAB_HISTORY[key] = h
        h["shown_count"] = int(h.get("shown_count", 0)) + 1
        h["last_shown"] = now

    def _add_phrase_rec(original_phrase: str, alternatives: List[str], reason: str, base_score: float):
        """Add a phrase-level recommendation."""
        original_phrase = (original_phrase or "").strip()
        if not original_phrase:
            return
        okey = original_phrase.lower()
        if okey in seen_originals:
            return
        cleaned = []
        seen_alt = set()
        for alt in alternatives:
            a = (alt or "").strip()
            if not a:
                continue
            k = a.lower()
            if k in seen_alt:
                continue
            if k == okey:
                continue
            if k in text_lower:
                continue
            if not _history_allows(a):
                continue
            seen_alt.add(k)
            cleaned.append(a)
            if len(cleaned) >= 3:
                break
        if not cleaned:
            return
        for a in cleaned:
            _record_shown(a)
        recommendations.append({
            "original": original_phrase, 
            "alternatives": cleaned, 
            "reason": reason, 
            "score": base_score,
            "difficulty_range": f"{min_diff:.1f}-{max_diff:.1f}",
            "user_level": user_profile.proficiency_level
        })
        seen_originals.add(okey)

    # Step 3: Personalized Word Recommendation
    
    # 3a. Check for words that need reinforcement (Spaced Repetition)
    reinforcement_words = adaptive_engine.select_words_for_reinforcement()
    for word in reinforcement_words:
        if word in text_lower:
            # User used a word that needed reinforcement - good!
            learning_repo.log_interaction(word, "used", "reinforcement practice")
    
    # 3b. Phrase-level recommendations for common patterns
    if "very good" in text_lower:
        if part == 3:
            _add_phrase_rec("very good", ["really solid", "genuinely impressive", "highly beneficial"], "More natural spoken phrasing", 50.0)
        else:
            _add_phrase_rec("very good", ["pretty decent", "really solid", "genuinely good"], "More natural spoken phrasing", 50.0)
        seen_words.add("good")

    if "very bad" in text_lower:
        _add_phrase_rec("very bad", ["pretty awful", "really poor", "quite disappointing"], "More natural spoken phrasing", 50.0)
        seen_words.add("bad")

    if "a lot" in text_lower:
        if part == 3:
            _add_phrase_rec("a lot", ["a great deal", "a substantial amount", "quite a bit"], "More natural quantity expression", 45.0)
        else:
            _add_phrase_rec("a lot", ["quite a bit", "loads of", "a bunch of"], "More natural quantity expression", 45.0)
        seen_words.add("lot")

    # 3c. Native collocation recommendations
    generic_adjs = {"good", "nice", "bad", "big", "great"}

    for i in range(len(doc) - 1):
        t1 = doc[i]
        t2 = doc[i + 1]
        if t1.is_space or t2.is_space:
            continue
        if t1.text.lower() == "very" and t2.pos_ == "ADJ":
            adj = t2.text.lower()
            advs = adv_by_adj.get(adj)
            if advs:
                alts = [f"{adv} {t2.text}" for adv in list(advs)]
                _add_phrase_rec(f"very {t2.text}", alts, "More native intensifier", 55.0)
        if t1.pos_ == "ADJ" and t2.pos_ in {"NOUN", "PROPN"}:
            noun = t2.text.lower()
            adjs = adj_by_noun.get(noun)
            if not adjs:
                continue
            current_adj = t1.text.lower()
            if current_adj in adjs:
                continue
            if current_adj not in generic_adjs and current_adj != "very":
                continue
            alts = []
            for a in sorted(adjs):
                phrase = f"{a} {t2.text}"
                if phrase.lower() in colloc_set:
                    alts.append(phrase)
            if not alts:
                for a in sorted(adjs):
                    alts.append(f"{a} {t2.text}")
            _add_phrase_rec(f"{t1.text} {t2.text}", alts, "More native collocation", 60.0)
            seen_words.add(current_adj)
    
    # 3d. Context-aware single word recommendations with AI Scoring
    global_contexts = detect_context(text)

    for token in doc: 
        word = token.text.lower()
        tag = token.tag_
        
        if word in CONTEXT_VOCAB and word not in seen_words:
            expected_pos = TARGET_POS_MAP.get(word, "")
            
            pos_match = False
            if expected_pos and tag.startswith(expected_pos):
                pos_match = True
            elif expected_pos == "JJ" and token.pos_ == "ADJ":
                pos_match = True
            elif expected_pos == "VB" and token.pos_ == "VERB":
                pos_match = True
            elif expected_pos == "NN" and token.pos_ == "NOUN":
                pos_match = True
            elif expected_pos == "RB" and token.pos_ == "ADV":
                pos_match = True
                
            if not pos_match and expected_pos:
                continue
                
            start = max(0, token.i - 4)
            end = min(len(doc), token.i + 5)
            window_span = doc[start:end]
            window_text = window_span.text
            
            local_contexts = detect_context(window_text)
            if not local_contexts:
                local_contexts = global_contexts
            
            options = CONTEXT_VOCAB[word]
            candidates = []
            
            matched_context = None
            for ctx in local_contexts:
                if ctx in options:
                    matched_context = ctx
                    for cand in options[ctx]:
                        cand_copy = cand.copy()
                        cand_copy["source"] = "context"
                        cand_copy["context"] = ctx
                        candidates.append(cand_copy)
            
            if not candidates:
                for cand in options.get("default", []):
                    cand_copy = cand.copy()
                    cand_copy["source"] = "default"
                    candidates.append(cand_copy)
            
            # Calculate context similarity for each candidate
            for cand in candidates:
                cand_word_raw = str(cand.get("word", "")).strip()
                if window_span.vector_norm:
                    cdoc = nlp(cand_word_raw)
                    if cdoc.vector_norm:
                        sim = float(np.dot(window_span.vector / window_span.vector_norm, 
                                          cdoc.vector / cdoc.vector_norm))
                        cand["context_similarity"] = sim
                    else:
                        cand["context_similarity"] = 0.0
                else:
                    cand["context_similarity"] = 0.0
            
            # Filter candidates based on register and part
            filtered_candidates = []
            for cand in candidates:
                cand_word_raw = str(cand.get("word", "")).strip()
                cand_key = cand_word_raw.lower()
                if not cand_key or cand_key == word:
                    continue
                reg = str(cand.get("reg", "neutral")).lower()
                if part in {1, 2} and reg in {"academic", "literary", "archaic", "formal"}:
                    continue
                if part == 3 and reg in {"archaic", "literary"}:
                    continue
                if cand_key in used_words:
                    continue
                if " " in cand_key and cand_key in text_lower:
                    continue
                    
                # Filter by difficulty appropriateness (i+1 principle)
                word_diff = cand.get("diff", 3)
                if word_diff < min_diff - 1 or word_diff > max_diff + 1:
                    continue  # Skip words too easy or too hard
                    
                # Check semantic similarity with original word
                ow = nlp(word)
                cdoc = nlp(cand_word_raw)
                if ow.vector_norm and cdoc.vector_norm:
                    sim_ow = float(np.dot(ow.vector / ow.vector_norm, cdoc.vector / cdoc.vector_norm))
                    if sim_ow < 0.35:
                        continue
                        
                if not _history_allows(cand_word_raw):
                    continue
                    
                filtered_candidates.append(cand)
            
            # Use AI Scoring System to rank candidates (Figure 2)
            ranked_candidates = scoring_system.rank_candidates(filtered_candidates, window_text)
            if trajectory_targets:
                boosted = []
                for score, cand in ranked_candidates:
                    ctext = str(cand.get("word", "")).strip().lower()
                    bonus = 8.0 if ctext in trajectory_targets else 0.0
                    boosted.append((score + bonus, cand))
                ranked_candidates = sorted(boosted, key=lambda x: x[0], reverse=True)
            
            best_alternatives = []
            seen_alts = set()
            
            for score, cand in ranked_candidates:
                cand_word_raw = str(cand.get("word", "")).strip()
                cand_key = cand_word_raw.lower()
                if not cand_key:
                    continue
                if cand_key not in seen_alts:
                    best_alternatives.append(cand_word_raw)
                    seen_alts.add(cand_key)
                if len(best_alternatives) >= 3:
                    break
            
            if not best_alternatives:
                continue
            
            top_cand = ranked_candidates[0][1] if ranked_candidates else {}
            top_reg = top_cand.get("reg", "neutral")
            top_diff = top_cand.get("diff", 3)

            if matched_context:
                reason = f"More natural when talking about {matched_context}"
            else:
                reason = "More natural in spoken English"
            if trajectory_targets and any(a.lower() in trajectory_targets for a in best_alternatives):
                reason += " — aligned with your current learning trajectory"

            if top_reg == "spoken":
                reason += " — commonly used in conversation"
            if top_diff >= 4:
                reason += " (advanced)"
            elif top_diff <= 2:
                reason += " (building foundation)"

            for alt in best_alternatives:
                _record_shown(alt)
            
            recommendations.append({
                "original": word,
                "alternatives": best_alternatives,
                "reason": reason,
                "score": ranked_candidates[0][0] if ranked_candidates else 0,
                "difficulty_range": f"{min_diff:.1f}-{max_diff:.1f}",
                "user_level": user_profile.proficiency_level
            })
            seen_words.add(word)

    ctx_rec = get_context_aware_word_recommendations(
        doc=doc,
        part=part,
        user_profile=user_profile,
        scoring_system=scoring_system,
        min_diff=min_diff,
        max_diff=max_diff,
        used_words=used_words,
        history_allows_fn=_history_allows,
        record_shown_fn=_record_shown,
        max_alternatives=5,
    )
    if isinstance(ctx_rec, dict):
        existing_originals = _build_existing_originals_set(recommendations)
        ctx_orig = _norm_rec_original(ctx_rec.get("original"))
        if ctx_orig and ctx_orig.lower() not in existing_originals:
            recommendations.append({
                "original": ctx_orig,
                "alternatives": _norm_rec_alternatives(ctx_rec.get("alternatives")),
                "reason": _norm_rec_reason(ctx_rec.get("reason")),
                "score": ctx_rec.get("score", 0),
                "difficulty_range": ctx_rec.get("difficulty_range", f"{min_diff:.1f}-{max_diff:.1f}"),
                "user_level": ctx_rec.get("user_level", user_profile.proficiency_level),
            })
    
    # Step 6: AI Model Refinement (Update proficiency estimate if needed)
    # This happens asynchronously based on accumulated data
    should_adjust, direction, magnitude = adaptive_engine.should_adjust_difficulty()
    if should_adjust:
        adaptive_engine.update_proficiency_estimate()
    
    # --- Context-Aware LLM Recommendations (WordRecom Integration) ---
    # Use DeepSeek to find semantically precise alternatives that fit the specific sentence structure.
    try:
        logger.info(f"[{LLM_MERGE_FIX_VERSION}] entering recommendation merge")
        llm_evaluator = get_llm_evaluator()
        if llm_evaluator and llm_evaluator.is_available():
            llm_recs = llm_evaluator.generate_contextual_vocabulary(
                text,
                f"Part {part}",
                user_band=user_profile.proficiency_level,
                trajectory_context=trajectory_context,
            )
            existing_originals = _build_existing_originals_set(recommendations)

            for rec in llm_recs if isinstance(llm_recs, list) else []:
                if not isinstance(rec, dict):
                    logger.warning(
                        f"[{LLM_MERGE_FIX_VERSION}] skip llm_rec non-dict type={type(rec).__name__}"
                    )
                    continue
                orig = _norm_rec_original(rec.get("original"))
                if not orig:
                    logger.warning(
                        f"[{LLM_MERGE_FIX_VERSION}] skip llm_rec empty original "
                        f"original_type={type(rec.get('original')).__name__} "
                        f"alternatives_type={type(rec.get('alternatives')).__name__} "
                        f"reason_type={type(rec.get('reason')).__name__}"
                    )
                    continue
                if orig.lower() in existing_originals:
                    continue

                recommendations.append({
                    "original": orig,
                    "alternatives": _norm_rec_alternatives(rec.get("alternatives")),
                    "reason": _norm_rec_reason(rec.get("reason")) + " (AI-Recommended)",
                    "score": 85.0,
                    "difficulty_range": f"{min_diff:.1f}-{max_diff:.1f}",
                    "user_level": user_profile.proficiency_level,
                })
                _safe_add_str_to_set(existing_originals, orig)

            word_recs = llm_evaluator.generate_contextual_word_recommendation(
                text,
                f"Part {part}",
                user_band=user_profile.proficiency_level,
                trajectory_context=trajectory_context,
            )

            for rec in word_recs if isinstance(word_recs, list) else []:
                if not isinstance(rec, dict):
                    logger.warning(
                        f"[{LLM_MERGE_FIX_VERSION}] skip word_rec non-dict type={type(rec).__name__}"
                    )
                    continue
                orig = _norm_rec_original(rec.get("original"))
                if not orig:
                    logger.warning(
                        f"[{LLM_MERGE_FIX_VERSION}] skip word_rec empty original "
                        f"original_type={type(rec.get('original')).__name__} "
                        f"alternatives_type={type(rec.get('alternatives')).__name__} "
                        f"reason_type={type(rec.get('reason')).__name__}"
                    )
                    continue
                if orig.lower() in existing_originals:
                    continue

                recommendations.append({
                    "original": orig,
                    "alternatives": _norm_rec_alternatives(rec.get("alternatives")),
                    "reason": _norm_rec_reason(rec.get("reason")) + " (AI-Recommended)",
                    "score": rec.get("score", 85.0),
                    "difficulty_range": rec.get("difficulty_range", f"{min_diff:.1f}-{max_diff:.1f}"),
                    "user_level": rec.get("user_level", user_profile.proficiency_level),
                })
                _safe_add_str_to_set(existing_originals, orig)

    except Exception as e:
        import traceback
        logger.error(
            f"[{LLM_MERGE_FIX_VERSION}] LLM recommendation merge failed: {e}; "
            f"recommendations_size={len(recommendations)} text_len={len(text)} part={part}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

    # Log session data
    learning_repo.log_session({
        "text_length": len(text),
        "part": part,
        "recommendations_count": len(recommendations),
        "words_analyzed": len(doc),
        "user_id": user_id,
        "trajectory_step": trajectory_context.get("trajectory_step"),
    })

    if ONTOLOGY_TRAJECTORY_ENABLED and learner_state is not None:
        learner_state.vocabulary_nodes = sorted(
            set(learner_state.vocabulary_nodes) | {w for w in trajectory_targets if w}
        )[:500]
        persist_learner_state(user_id)
                
    return recommendations

def _fallback_model_answer(transcript: str, part: int, user_band: float = 6.0) -> str:
    """Rule-based polish when LLM is unavailable or returns empty."""
    if not transcript or len(transcript.strip()) < 5:
        return "Transcript too short for a model answer. Try a longer response."
    t = transcript.strip()
    t = re.sub(r"\s+", " ", t)
    if t:
        t = t[0].upper() + t[1:]
    target_band = min(9.0, user_band + 1.0)
    path = _get_local_model_path()
    hint = (
        f"Band {target_band:.1f} model unavailable. Ensure {path} exists, or set DEEPSEEK_API_KEY. Install peft: pip install peft\n\n"
        "Quick polish of your response:\n\n"
    )
    return hint + t


# Add new function to generate model answer only
def get_model_answer(text: str, part: int, user_band: float = 6.0, trajectory_context: Optional[Dict[str, Any]] = None):
    try:
        llm_evaluator = get_llm_evaluator()
        if llm_evaluator and llm_evaluator.is_available():
            return llm_evaluator.generate_model_answer(
                text,
                f"Part {part}",
                user_band=user_band,
                trajectory_context=trajectory_context or {},
            )
    except Exception as e:
        logger.error(f"Model answer generation failed: {e}")
    return ""

@app.post("/feedback_recommendation")
async def feedback_recommendation(feedback: FeedbackRequest):
    """
    Handle user feedback on word recommendations.
    Step 4 in the workflow: Interactive Exercises & Feedback
    """
    word = (feedback.recommended or "").strip().lower()
    action = (feedback.action or "").strip().lower()
    user_id = (feedback.user_id or "default").strip() or "default"
    
    if not word:
        raise HTTPException(status_code=400, detail="recommended is required")
    if action not in {"accept", "ignore"}:
        raise HTTPException(status_code=400, detail="action must be 'accept' or 'ignore'")

    # Get learning repository for this user
    learning_repo = get_learning_repo(user_id)
    user_profile = get_user_profile(user_id)
    
    # Log interaction to learning repository (Step 5: Performance Logged)
    learning_repo.log_interaction(word, action, feedback.original or "")
    
    # Update vocabulary tracking
    if action == "accept":
        user_profile.vocabulary_learning.add(word)
        if word in user_profile.vocabulary_introduced:
            user_profile.vocabulary_introduced.discard(word)
    if ONTOLOGY_TRAJECTORY_ENABLED:
        state = get_learner_state(user_id)
        if state is not None:
            prev = float(state.concept_mastery.get(word, 0.0))
            delta = 0.15 if action == "accept" else -0.08
            state.concept_mastery[word] = max(0.0, min(1.0, prev + delta))
            if state.concept_mastery[word] >= 0.75:
                if word not in state.mastered_nodes:
                    state.mastered_nodes.append(word)
            persist_learner_state(user_id)
    
    # Legacy tracking for backwards compatibility
    h = USER_VOCAB_HISTORY.get(word)
    if not h:
        h = {"shown_count": 0, "accepted_count": 0, "ignored_count": 0, "last_shown": 0.0}
        USER_VOCAB_HISTORY[word] = h

    if action == "accept":
        h["accepted_count"] = int(h.get("accepted_count", 0)) + 1
    else:
        h["ignored_count"] = int(h.get("ignored_count", 0)) + 1

    # Get updated stats from learning repository
    stats = learning_repo.get_word_stats(word)
    
    return {
        "status": "success", 
        "user_id": user_id,
        "history": h,
        "learning_stats": {
            "mastery_level": stats.get("mastery_level", 0),
            "retention_score": stats.get("retention_score", 0),
            "total_interactions": stats.get("interaction_count", 0)
        }
    }


# ============================================================================
# API ENDPOINTS FOR USER PROFILE AND PROGRESS MONITORING
# (Figure 4: Progress Monitoring Dashboard)
# ============================================================================

class UserProfileRequest(BaseModel):
    user_id: str = "default"
    proficiency_level: Optional[float] = None
    target_level: Optional[float] = None
    learning_goals: Optional[List[str]] = None
    preferred_topics: Optional[List[str]] = None
    register_preference: Optional[str] = None


@app.get("/user/profile/{user_id}")
async def get_profile(user_id: str = "default"):
    """
    Get user profile for personalized learning.
    Step 1: User Profile Creation (read)
    """
    profile = get_user_profile(user_id)
    learner_state = get_learner_state(user_id) if ONTOLOGY_TRAJECTORY_ENABLED else None
    trajectory = []
    current_step = None
    if learner_state is not None:
        trajectory = learner_state.trajectory_plan[:5]
        if trajectory and 0 <= learner_state.trajectory_step < len(trajectory):
            current_step = trajectory[learner_state.trajectory_step]
    return {
        "status": "success",
        "profile": profile.to_dict(),
        "current_trajectory_step": current_step,
        "next_trajectory_steps": trajectory[:3],
        "concept_mastery_summary": (learner_state.concept_mastery if learner_state else {}),
    }


@app.post("/user/profile")
async def update_profile(request: UserProfileRequest):
    """
    Update user profile settings.
    Step 1: User Profile Creation (write)
    """
    profile = get_user_profile(request.user_id)
    
    if request.proficiency_level is not None:
        profile.proficiency_level = max(1.0, min(9.0, request.proficiency_level))
    if request.target_level is not None:
        profile.target_level = max(1.0, min(9.0, request.target_level))
    if request.learning_goals is not None:
        profile.learning_goals = request.learning_goals
    if request.preferred_topics is not None:
        profile.preferred_topics = request.preferred_topics
    if request.register_preference is not None:
        profile.register_preference = request.register_preference
        
    return {
        "status": "success",
        "profile": profile.to_dict()
    }


@app.get("/user/progress/{user_id}")
async def get_progress(user_id: str = "default"):
    """
    Get learning progress and analytics.
    Step 7: Progress Monitoring Dashboard
    """
    profile = get_user_profile(user_id)
    learning_repo = get_learning_repo(user_id)
    adaptive_engine = AdaptiveLearningEngine(profile, learning_repo)
    
    # Get performance analysis
    performance = adaptive_engine.analyze_user_performance()
    
    # Get difficulty recommendation
    min_diff, optimal_diff, max_diff = adaptive_engine.get_recommended_difficulty_range()
    
    # Get words needing reinforcement
    reinforcement_words = adaptive_engine.select_words_for_reinforcement()
    
    # Calculate vocabulary growth
    vocab_stats = {
        "total_introduced": len(profile.vocabulary_introduced),
        "currently_learning": len(profile.vocabulary_learning),
        "mastered": len(profile.vocabulary_mastered),
        "total_unique_words": len(learning_repo.engagement_metrics)
    }
    
    # Mastery distribution
    mastery_dist = {
        "new": 0,
        "introduced": 0,
        "learning": 0,
        "learned": 0,
        "mastered": 0
    }
    for metrics in learning_repo.engagement_metrics.values():
        level = metrics.get("mastery_level", 0)
        if level == 0:
            mastery_dist["new"] += 1
        elif level == 1:
            mastery_dist["introduced"] += 1
        elif level == 2:
            mastery_dist["learning"] += 1
        elif level == 3:
            mastery_dist["learned"] += 1
        elif level >= 4:
            mastery_dist["mastered"] += 1
    
    learner_state = get_learner_state(user_id) if ONTOLOGY_TRAJECTORY_ENABLED else None
    next_steps = []
    current_step = None
    concept_summary = {}
    if learner_state is not None:
        next_steps = learner_state.trajectory_plan[:3]
        if learner_state.trajectory_plan and 0 <= learner_state.trajectory_step < len(learner_state.trajectory_plan):
            current_step = learner_state.trajectory_plan[learner_state.trajectory_step]
        concept_summary = learner_state.concept_mastery

    return {
        "status": "success",
        "user_id": user_id,
        "current_level": profile.proficiency_level,
        "target_level": profile.target_level,
        "sessions_completed": profile.total_sessions,
        "performance_metrics": performance,
        "vocabulary_stats": vocab_stats,
        "mastery_distribution": mastery_dist,
        "recommended_difficulty": {
            "min": min_diff,
            "optimal": optimal_diff,
            "max": max_diff
        },
        "current_trajectory_step": current_step,
        "next_trajectory_steps": next_steps,
        "concept_mastery_summary": concept_summary,
        "words_to_review": reinforcement_words,
        "proficiency_history": learning_repo.proficiency_history[-10:]  # Last 10 updates
    }


@app.get("/user/vocabulary/{user_id}")
async def get_vocabulary_stats(user_id: str = "default"):
    """
    Get detailed vocabulary statistics and word-level data.
    """
    learning_repo = get_learning_repo(user_id)
    
    word_stats = []
    for word, metrics in learning_repo.engagement_metrics.items():
        retention = learning_repo.calculate_retention_score(word)
        word_stats.append({
            "word": word,
            "mastery_level": metrics.get("mastery_level", 0),
            "retention_score": round(retention, 2),
            "shown_count": metrics.get("shown_count", 0),
            "accepted_count": metrics.get("accepted_count", 0),
            "used_count": metrics.get("used_in_speech_count", 0),
            "first_seen": metrics.get("first_seen"),
            "last_interaction": metrics.get("last_interaction")
        })
    
    # Sort by mastery level (learning words first)
    word_stats.sort(key=lambda x: (x["mastery_level"], -x["retention_score"]))
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_words": len(word_stats),
        "words": word_stats
    }


@app.post("/user/proficiency/{user_id}")
async def update_proficiency(user_id: str = "default", scores: Dict[str, float] = None):
    """
    Update user proficiency based on evaluation scores.
    Step 6: AI Model Refinement
    """
    profile = get_user_profile(user_id)
    learning_repo = get_learning_repo(user_id)
    adaptive_engine = AdaptiveLearningEngine(profile, learning_repo)
    
    previous_level = profile.proficiency_level
    new_level = adaptive_engine.update_proficiency_estimate(scores)
    
    return {
        "status": "success",
        "user_id": user_id,
        "previous_level": previous_level,
        "new_level": new_level,
        "adjustment_reason": "Based on evaluation scores and learning performance"
    }


class WordUsageRequest(BaseModel):
    user_id: str = "default"
    word: str
    context: str = ""


@app.post("/user/word_used")
async def log_word_usage(request: WordUsageRequest):
    """
    Log when a user successfully uses a recommended word.
    This helps track vocabulary acquisition and retention.
    """
    learning_repo = get_learning_repo(request.user_id)
    profile = get_user_profile(request.user_id)
    
    word = request.word.strip().lower()
    if not word:
        raise HTTPException(status_code=400, detail="word is required")
    
    # Log the usage
    learning_repo.log_interaction(word, "used", request.context)
    if ONTOLOGY_TRAJECTORY_ENABLED:
        state = get_learner_state(request.user_id)
        if state is not None:
            prev = float(state.concept_mastery.get(word, 0.0))
            state.concept_mastery[word] = max(0.0, min(1.0, prev + 0.2))
            if state.concept_mastery[word] >= 0.75 and word not in state.mastered_nodes:
                state.mastered_nodes.append(word)
            persist_learner_state(request.user_id)
    
    # Update vocabulary tracking
    if word in profile.vocabulary_learning:
        # Check if mastered
        stats = learning_repo.get_word_stats(word)
        if stats.get("mastery_level", 0) >= 4:
            profile.vocabulary_mastered.add(word)
            profile.vocabulary_learning.discard(word)
    
    stats = learning_repo.get_word_stats(word)
    
    return {
        "status": "success",
        "user_id": request.user_id,
        "word": word,
        "mastery_level": stats.get("mastery_level", 0),
        "retention_score": stats.get("retention_score", 0),
        "message": f"Word '{word}' usage logged successfully"
    }

SEMANTIC_CLUSTERS = {
    "Part1": [
        # Work / Studies
        {"band": 9.0, "topic": "work_study", "text": "I’m currently juggling my studies with a part-time job, which can be quite hectic, but I find it rewarding. It gives me a sense of financial independence and allows me to apply what I learn in a practical setting."},
        {"band": 8.0, "topic": "work_study", "text": "I work as a software engineer. It’s a challenging role, but I enjoy problem-solving. Sometimes the deadlines are tight, which is stressful, but on the whole, I’m satisfied with my career path."},
        {"band": 6.0, "topic": "work_study", "text": "I am a student. I study business. It is good. I like my school because it is big and has many people."},
        # Hobbies / Free Time
        {"band": 9.0, "topic": "hobbies", "text": "I’m quite partial to outdoor activities, particularly hiking. There’s something therapeutic about being in nature that helps me disconnect from the daily grind. If the weather permits, I’ll usually head to the mountains every other weekend."},
        {"band": 8.0, "topic": "hobbies", "text": "In my free time, I usually go to the gym or read books. I think it’s important to stay active. Also, reading helps me relax after a long day at work."},
        {"band": 5.0, "topic": "hobbies", "text": "I like play football. It is fun. I play with my friend. We go to park. I like it very much."},
        # Hometown / Accommodation
        {"band": 9.0, "topic": "home", "text": "I reside in a bustling metropolitan area, which has its pros and cons. While the amenities are top-notch and everything is accessible, the noise pollution can be a bit overwhelming at times. Ideally, I’d prefer somewhere a bit more tranquil."},
        {"band": 7.0, "topic": "home", "text": "I live in an apartment in the city center. It’s very convenient because it’s close to shops and restaurants. However, it can be noisy at night, which is a bit annoying."},
    ],
    "Part2": [
        # Describe a person/event/object
        {"band": 9.0, "topic": "narrative", "text": "One event that vividly stands out in my memory is a trip I took to Japan. The cultural immersion was unlike anything I’d experienced before. From the meticulous attention to detail in their architecture to the profound sense of respect in their social interactions, it was truly eye-opening."},
        {"band": 8.0, "topic": "narrative", "text": "I’d like to talk about my grandfather. He was a very influential figure in my life. He taught me the value of hard work and integrity. I remember he used to tell me stories about his youth, which were always fascinating."},
        {"band": 6.0, "topic": "narrative", "text": "I want to talk about my holiday. I went to the beach. It was beautiful. The water was blue. We ate seafood. It was a very happy time for me and my family."},
    ],
    "Part3": [
        # Abstract / Society / Technology
        {"band": 9.0, "topic": "abstract", "text": "From my perspective, technological advancement is a double-edged sword. On one hand, it has revolutionized how we communicate, bridging geographical gaps instantly. On the other hand, it has fostered a sense of isolation, as face-to-face interactions are increasingly replaced by digital ones."},
        {"band": 8.0, "topic": "abstract", "text": "I think technology has changed our lives a lot. It makes things faster and easier. For example, we can talk to people on the other side of the world. But some people spend too much time on their phones, which is not good for their social skills."},
        {"band": 6.0, "topic": "abstract", "text": "Technology is important. It helps us do many things. Computers and phones are useful. But sometimes it is bad because people play games all day."},
    ]
}

_semantic_cluster_cache = None

def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec)
    if denom == 0:
        return vec
    return vec / denom

def _softmax(x: np.ndarray) -> np.ndarray:
    # Stable softmax
    x = x - np.max(x)
    e = np.exp(x)
    denom = np.sum(e)
    if denom == 0:
        return np.ones_like(x) / max(1, len(x))
    return e / denom

def _get_semantic_cluster_vectors():
    global _semantic_cluster_cache
    if _semantic_cluster_cache is not None:
        return _semantic_cluster_cache
        
    nlp = get_spacy()
    clusters = []
    
    # Iterate through Parts (Part1, Part2, etc.)
    for part_name, anchors in SEMANTIC_CLUSTERS.items():
        for a in anchors:
            d = nlp(a["text"])
            if not d.vector_norm:
                continue
            clusters.append({
                "part": part_name,
                "topic": a["topic"],
                "band": float(a["band"]),
                "vec": _normalize_vector(d.vector),
                "text_preview": a["text"][:30] + "..."
            })
            
    _semantic_cluster_cache = clusters
    return clusters

def _semantic_transfer_band(doc) -> tuple[Optional[float], float, str]:
    """
    Returns: (predicted_band, alpha_weight, rationale_str)
    """
    if not doc.vector_norm:
        return None, 0.0, ""
        
    anchors = _get_semantic_cluster_vectors()
    if not anchors:
        return None, 0.0, ""
        
    v = _normalize_vector(doc.vector)
    
    # Calculate similarities to all anchors
    sims = np.array([float(np.dot(v, a["vec"])) for a in anchors], dtype=np.float32)
    
    # Adaptive Temperature based on "uncertainty"
    # If max similarity is low, we are uncertain -> lower temperature (sharpen) or just trust less?
    # Actually, we want alpha to depend on max_sim.
    max_sim = np.max(sims)
    
    # Softmax for weighted aggregation
    # Scale factor 8.0 makes the softmax sharper around the best match
    weights = _softmax(sims * 8.0)
    
    # Weighted average for band prediction
    pred_band = float(np.sum(weights * np.array([a["band"] for a in anchors], dtype=np.float32)))
    
    # Identify best match for rationale
    best_idx = np.argmax(sims)
    best_match = anchors[best_idx]
    
    # --- Adaptive Alpha Calculation ---
    # Logic: 
    # 1. High max_sim (>0.85) -> User is very close to a known anchor -> High confidence -> Alpha ~ 0.5-0.6
    # 2. Moderate max_sim (0.7-0.85) -> Some drift -> Alpha ~ 0.3-0.5
    # 3. Low max_sim (<0.7) -> Off-topic or outlier -> Low confidence -> Alpha < 0.2 (Rely on grammar/vocab)
    
    # Sigmoid-like curve for alpha
    # Center at 0.75, steepness 10
    # alpha = 0.6 / (1 + exp(-15 * (max_sim - 0.75)))
    # Simplified piecewise linear:
    if max_sim > 0.85:
        alpha = 0.6
    elif max_sim > 0.75:
        alpha = 0.4
    elif max_sim > 0.65:
        alpha = 0.2
    else:
        alpha = 0.05
        
    rationale = f"Matched '{best_match['topic']}' style ({best_match['part']}, Sim: {max_sim:.2f})"
    
    return pred_band, alpha, rationale

def _ensure_wav_for_loading(audio_path: str) -> tuple:
    """
    Convert to WAV if needed to avoid libmpg123 errors (non-standard MP3/AAC from mobile).
    Returns (path_to_load, was_temp_created).
    """
    ext = (os.path.splitext(audio_path)[1] or "").lower()
    if ext in (".wav",):
        try:
            librosa.load(audio_path, sr=16000, duration=0.1)
            return audio_path, False
        except Exception:
            pass
    # Convert with ffmpeg for mp3/m4a/aac or when direct load fails
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path, True
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"ffmpeg conversion failed: {e}")
    # Fallback: return original (may fail with libmpg123)
    return audio_path, False


def process_audio_signal(input_path: str):
    """
    Applies Digital Signal Processing (DSP) to improve audio quality and accuracy.
    Ref: Audio Visualization (44.1kHz, High Dynamic Range)
    1. Resampling to 44.1kHz (preserve high frequency details up to 20kHz)
    2. Silence Trimming (relaxed threshold to avoid omitting soft speech)
    3. Normalization (ensure consistent volume)
    Returns: (processed_file_path, duration_in_seconds)
    """
    load_path, wav_created = _ensure_wav_for_loading(input_path)
    try:
        y, sr = librosa.load(load_path, sr=16000)
        
        y_trimmed, _ = librosa.effects.trim(y, top_db=60)
        y_pre = librosa.effects.preemphasis(y_trimmed, coef=0.97)
        
        if np.max(np.abs(y_pre)) > 0:
            y_norm = y_pre / np.max(np.abs(y_pre))
        else:
            y_norm = y_pre
            
        # Calculate duration
        duration = len(y_norm) / sr
        
        # Save processed temp file
        fd, temp_processed = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_processed, y_norm, sr)
        
        return temp_processed, duration
    except Exception as e:
        logger.warning(f"DSP Error: {e}")
        return input_path, 0.0
    finally:
        if wav_created and os.path.exists(load_path):
            try:
                os.unlink(load_path)
            except OSError:
                pass

def calculate_score_with_rationale(text: str, duration: float, pron_score: float, part: int = 1, temporal: Dict = None):
    # Pre-processing with spaCy
    nlp = get_spacy()
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_punct]
    
    if not words:
        return 5.0, {}, ["Not enough speech detected."]
    
    rationale = []
    scores = {}
    
    # --- PART-SPECIFIC TUNING PARAMETERS ---
    # Part 1: Introduction (Direct answers, length ~20-40 words ideal, simple is okay)
    # Part 2: Long Turn (Sustained speech, length > 100 words ideal, coherence focus)
    # Part 3: Discussion (Abstract/Complex, length ~40-60 words ideal, logic focus)
    
    part_rationale = f"Mode: Part {part} - "
    if part == 1:
        part_rationale += "Introduction/Interview"
    elif part == 2:
        part_rationale += "Long Turn"
    else:
        part_rationale += "Discussion"
    rationale.append(f"ℹ️ **{part_rationale}**")
    
    # --- Auditory/Acoustic Feature Calibration (Gammatone/MFCC) ---
    # Check for "High-Pitch Voice Details" via MFCC/Spectral features if available
    # High-pitch detail retention implies better intonation analysis potential
    if temporal and "mfcc_mean" in temporal:
         # Check if high-frequency energy is preserved (Speech Band Ratio)
         sber = temporal.get("speech_band_energy_ratio", 0.0)
         if sber > 0.4:
             rationale.append("✅ **Audio Quality**: High-pitch details preserved (Gammatone-aligned spectrum).")
         elif sber < 0.1:
             rationale.append("⚠️ **Audio Quality**: High-frequency details may be attenuated.")

    # --- SEMANTIC ANALYSIS METRICS ---
    
    # 1. Semantic Coherence (Flow between sentences)
    sentences = list(doc.sents)
    coherence_scores = []
    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            # Check similarity if both sentences have vectors
            if sentences[i].vector_norm and sentences[i+1].vector_norm:
                coherence_scores.append(sentences[i].similarity(sentences[i+1]))
    
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    # Boost coherence if it's very high (e.g. staying on topic)
    # But punish if it's too high (repetitive)? No, usually high similarity means good flow in this context.
    
    # 2. Semantic Lexical Variety (Unique Lemmas of Content Words)
    content_words = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop]
    unique_content_lemmas = set(content_words)
    semantic_variety_ratio = len(unique_content_lemmas) / max(30, len(content_words)) if content_words else 0.0

    # 3. Dependency-Based Complexity
    # Check for passive voice, relative clauses via dependency labels
    has_passive = any(token.dep_ == "auxpass" for token in doc)
    has_relative = any(token.dep_ in ["relcl", "acl"] for token in doc)
    has_subordinate = any(token.dep_ in ["mark", "advcl"] for token in doc)
    
    # ---------------------------------
    
    # 0. Contextual & Semantic Analysis
    # Lexical Bundles Analysis (Based on Biber 2004 Categories)
    bundle_count = 0
    stance_bundle_count = 0
    discourse_bundle_count = 0
    referential_bundle_count = 0
    
    text_lower = text.lower()
    
    # Check Stance Bundles (High reward for Spoken)
    for bundle in SPOKEN_STANCE_BUNDLES:
        if bundle.lower() in text_lower:
            stance_bundle_count += 1
            bundle_count += 1
            
    # Check Discourse Bundles (High reward for Spoken)
    for bundle in SPOKEN_DISCOURSE_BUNDLES:
        if bundle.lower() in text_lower:
            discourse_bundle_count += 1
            bundle_count += 1
            
    # Check Referential Bundles (Moderate reward, good for academic LR but avoid overuse)
    for bundle in REFERENTIAL_BUNDLES:
        if bundle.lower() in text_lower:
            referential_bundle_count += 1
            bundle_count += 1
            
    # Check ASWL Usage (Part 3 Specific)
    aswl_hits = 0
    aswl_unique_words = set()
    if part == 3:
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in ASWL_VOCAB:
                aswl_hits += 1
                aswl_unique_words.add(lemma)
            
    # Intonation Indicators
    punctuation_variety = (text.count('?') + text.count('!')) > 0
    attitude_count = sum(1 for m in ATTITUDE_MARKERS if m in text_lower)
            
    # Sentiment / Subjectivity
    try:
        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity 
    except Exception:
        subjectivity = 0.5 
    
    # 1. Fluency & Coherence (FC)
    # RECALIBRATED to align with Official IELTS Band Descriptors
    # Band 9: "Fluent with only very occasional repetition/self-correction"
    #         "Hesitation only to prepare content, NOT to find words/grammar"
    # Band 8: "Hesitation may occasionally be for words/grammar, but MOST content-related"
    # Band 7: "Some hesitation/repetition/self-correction, often mid-sentence"
    # Band 6: "Coherence may be LOST at times due to hesitation/repetition"
    # Band 5: "Relies on repetition/self-correction and/or slow speech"
    
    wpm = (len(words) / duration) * 60 if duration > 0 else 0
    
    # Count Discourse Markers (Case insensitive)
    discourse_count = 0
    for marker in DISCOURSE_MARKERS:
        if re.search(r'\b' + re.escape(marker) + r'\b', text_lower):
            discourse_count += 1
            
    # Count Fillers
    filler_count = 0
    for f in FILLERS:
        filler_count += len(re.findall(r'\b' + re.escape(f) + r'\b', text_lower))
    filler_density = filler_count / max(1, len(words))

    fc_score = 4.0
    
    # STRICTER: Reduced bonuses
    filler_limit_bonus = 0.0
    if wpm > 140:
        filler_limit_bonus = 0.01  # Reduced from 0.02

    # Semantic Coherence: Only boost if very high
    coherence_bonus = 0.01 if avg_coherence > 0.7 else 0.0  # Reduced from 0.03
    
    # Dynamic Discourse Threshold: STRICTER requirements
    base_req_discourse = max(2, int(len(words) / 30))  # Increased from /35
    if part == 1:
        req_discourse = max(1, base_req_discourse - 1)
    elif part == 2:
        req_discourse = max(3, base_req_discourse)  # Increased from 2
    else:  # Part 3
        req_discourse = max(3, base_req_discourse)  # Increased from 2

    # STRICTER WPM Thresholds (Official descriptors emphasize "fluent" speech)
    # Native speakers typically speak 120-150 WPM in conversation
    min_wpm_9 = 130  # Increased from 110 - "Fluent with only very occasional..."
    min_wpm_8 = 115  # Increased from 100
    min_wpm_7 = 100  # Increased from 90 - "Keep going without noticeable effort"
    min_wpm_6 = 85   # Increased from 75 - "Willingness to produce long turns"
    min_wpm_5 = 70   # Increased from 60
    
    # Part 3 allows slightly lower WPM (thinking time for abstract topics)
    if part == 3:
        min_wpm_9 = 115
        min_wpm_8 = 100
        min_wpm_7 = 90
        min_wpm_6 = 75
    
    # STRICTER Filler Density Thresholds
    # Band 9: "only very occasional" means < 2%
    # Band 8: "occasional" means < 4%
    max_filler_9 = 0.02 + filler_limit_bonus + coherence_bonus  # Reduced from 0.04
    max_filler_8 = 0.04 + filler_limit_bonus + coherence_bonus  # Reduced from 0.08
    max_filler_7 = 0.07 + filler_limit_bonus + coherence_bonus  # Reduced from 0.12
    max_filler_6 = 0.10  # Reduced from 0.15

    if wpm > min_wpm_9 and filler_density < max_filler_9 and discourse_count >= req_discourse and avg_coherence > 0.6:
        fc_score = 9.0
        rationale.append("✅ **Fluency**: Band 9 - Fluent; hesitation used only to prepare content.")
    elif wpm > min_wpm_8 and filler_density < max_filler_8 and discourse_count >= max(2, req_discourse - 1) and avg_coherence > 0.5:
        fc_score = 8.0
        rationale.append("✅ **Fluency**: Band 8 - Fluent; hesitation mostly content-related, occasionally for words.")
    elif wpm > min_wpm_7 and filler_density < max_filler_7 and discourse_count >= 2:
        fc_score = 7.0
        rationale.append("✅ **Fluency**: Band 7 - Able to keep going without noticeable effort.")
    elif wpm > min_wpm_6 and filler_density < max_filler_6:
        fc_score = 6.0
        rationale.append("⚠️ **Fluency**: Band 6 - Willing to produce long turns but loses coherence at times.")
    elif wpm > min_wpm_5:
        fc_score = 5.0
        rationale.append("⚠️ **Fluency**: Band 5 - Relies on repetition and self-correction.")
    else:
        fc_score = 4.0
        rationale.append("❌ **Fluency**: Band 4 - Unable to keep going without noticeable pauses.")
        
    # Part 2 Length Penalty (Must be long enough)
    if part == 2 and len(words) < 80:
        if fc_score > 5.0:
            fc_score -= 1.0
            rationale.append("⚠️ **Fluency**: Part 2 response too short (< 80 words). Sustain speech longer.")

    if part == 3 and temporal:
        stats = get_fcael_stats().get("part3", {}) if get_fcael_stats else {}
        pause_ratio = float(temporal.get("pause_ratio", 0.0))
        mean_pause_s = float(temporal.get("mean_pause_s", 0.0))
        articulation_wpm = temporal.get("articulation_wpm")
        adj = 0.0
        if stats:
            pr = stats.get("pause_ratio", {})
            mp = stats.get("mean_pause_s", {})
            aw = stats.get("articulation_wpm", {})
            pr_p50 = float(pr.get("p50", 0.25) or 0.25)
            pr_p90 = float(pr.get("p90", 0.40) or 0.40)
            if pause_ratio > pr_p90:
                adj -= 0.5
            elif pause_ratio < pr_p50:
                adj += 0.2
            mp_p50 = float(mp.get("p50", 0.55) or 0.55)
            mp_p90 = float(mp.get("p90", 0.90) or 0.90)
            if mean_pause_s > mp_p90:
                adj -= 0.3
            elif mean_pause_s < mp_p50 and mean_pause_s > 0:
                adj += 0.1
            if isinstance(articulation_wpm, (int, float)):
                aw_p10 = float(aw.get("p10", 105) or 105)
                aw_p50 = float(aw.get("p50", 140) or 140)
                aw_p90 = float(aw.get("p90", 175) or 175)
                if articulation_wpm > aw_p90:
                    adj -= 0.2
                elif articulation_wpm >= aw_p50:
                    adj += 0.2
                elif articulation_wpm < aw_p10:
                    adj -= 0.2
        else:
            if pause_ratio > 0.40:
                adj -= 0.5
            elif pause_ratio < 0.22:
                adj += 0.2
            if mean_pause_s > 0.9:
                adj -= 0.3
        if adj != 0.0:
            fc_score = max(4.0, min(9.0, fc_score + adj))
            rationale.append(f"Temporal fluency adjustment (Part 3): {adj:+.1f} (pause_ratio={pause_ratio:.2f}, mean_pause={mean_pause_s:.2f}s)")

    if avg_coherence < 0.3 and fc_score > 6.0:
        fc_score -= 1.0
        rationale.append("⚠️ **Coherence**: Ideas seem disjointed (Low semantic similarity).")

    # Coherence/attitude bonus: only for truly outstanding flow, and capped tighter
    if fc_score >= 7.0 and fc_score < 9.0 and subjectivity > 0.7 and avg_coherence > 0.75 and attitude_count >= 2:
        fc_score += 0.5
        rationale.append("✅ **Fluency**: Exceptional coherence and attitudinal markers.")
         
    scores['fluency'] = fc_score

    # 2. Lexical Resource (LR)
    # RECALIBRATED to align with Official IELTS Band Descriptors
    # Band 9: "Total flexibility and precise use in ALL contexts"
    #         "Sustained use of accurate and IDIOMATIC language"
    # Band 8: "Wide resource, readily and flexibly used"
    #         "Skillful use of less common and idiomatic items"
    # Band 7: "Resource flexibly used to discuss a variety of topics"
    #         "Some ability to use less common and idiomatic items"
    # Band 6: "Resource sufficient to discuss topics at length"
    #         "Vocabulary use may be inappropriate but meaning is clear"
    # Band 5: "Resource sufficient for familiar topics but LIMITED flexibility"
    
    # Use Semantic Variety instead of just length
    advanced_words = [w for w in words if len(w) > 5]
    lex_ratio = len(advanced_words) / max(1, len(words))
    
    idiom_count = bundle_count 
    
    # STRICTER: Dynamic Idiom Threshold - require more idiomatic expressions
    req_idioms = max(2, int(len(words) / 50))  # Increased from /60

    lr_score = 4.0
    
    # Weighted Bundle Score for LR
    weighted_bundle_score = (stance_bundle_count * 1.5) + (discourse_bundle_count * 1.5) + (referential_bundle_count * 1.0)
    
    # STRICTER Thresholds based on official descriptors
    # Band 9 requires "total flexibility" and "sustained idiomatic" - very high bar
    # Band 8 requires "wide resource" with "skillful" idiomatic use
    
    # Adjust thresholds based on Part
    if part == 3:
        # Part 3: "Total flexibility" requires abstract vocabulary
        lex_thresh_9 = 0.20   # Increased from 0.15
        sem_thresh_9 = 0.75   # Increased from 0.70
        lex_thresh_8 = 0.16
        sem_thresh_8 = 0.65
        
        # ASWL (Academic Spoken Word List) requirements for Part 3
        aswl_density = aswl_hits / max(1, len(words))
        if aswl_density > 0.06:
            weighted_bundle_score += 1.5
            rationale.append(f"✅ **Academic Vocab**: Strong abstract vocabulary ({len(aswl_unique_words)} unique).")
        elif aswl_density > 0.04:
            weighted_bundle_score += 0.5
            rationale.append("✅ **Academic Vocab**: Good use of abstract vocabulary.")
        elif aswl_density < 0.02:
            rationale.append("⚠️ **Part 3 Tip**: Use more abstract/academic vocabulary for discussion.")
            
    elif part == 1:
        # Part 1: Can be simpler but still needs variety
        lex_thresh_9 = 0.15   # Increased from 0.11
        sem_thresh_9 = 0.65   # Increased from 0.60
        lex_thresh_8 = 0.12
        sem_thresh_8 = 0.55
    else:  # Part 2
        lex_thresh_9 = 0.18   # Increased from 0.13
        sem_thresh_9 = 0.70   # Increased from 0.65
        lex_thresh_8 = 0.14
        sem_thresh_8 = 0.60

    # STRICTER scoring logic
    # Band 9: Must have BOTH high lexical ratio AND strong idiomatic usage
    if (lex_ratio > lex_thresh_9 and semantic_variety_ratio > sem_thresh_9) and weighted_bundle_score >= req_idioms + 1:
        lr_score = 9.0
        rationale.append("✅ **Lexical**: Band 9 - Total flexibility and precise use in all contexts.")
    elif (lex_ratio > lex_thresh_8 or semantic_variety_ratio > sem_thresh_8) and weighted_bundle_score >= req_idioms:
        lr_score = 8.0
        rationale.append("✅ **Lexical**: Band 8 - Wide resource, readily and flexibly used.")
    elif (lex_ratio > 0.11 or semantic_variety_ratio > 0.50) and weighted_bundle_score >= max(1, req_idioms - 1):
        lr_score = 7.0
        rationale.append("✅ **Lexical**: Band 7 - Resource flexibly used to discuss a variety of topics.")
    elif lex_ratio > 0.09 or semantic_variety_ratio > 0.40:
        lr_score = 6.0
        rationale.append("⚠️ **Lexical**: Band 6 - Resource sufficient to discuss topics at length.")
    elif lex_ratio > 0.06:
        lr_score = 5.0
        rationale.append("⚠️ **Lexical**: Band 5 - Limited flexibility; basic meaning conveyed.")
    else:
        lr_score = 4.0
        rationale.append("❌ **Lexical**: Band 4 - Resource sufficient for familiar topics only.")
        
    # Check for "Bookish" penalty (Too many referential bundles, no stance/discourse)
    if referential_bundle_count > 2 and (stance_bundle_count + discourse_bundle_count) == 0:
        if lr_score > 6.0:
            lr_score -= 0.5
            rationale.append("⚠️ **Style**: Tone may be too academic/written. Use more spoken stance markers.")
        
    # 3. Collocation & Colligation Analysis (Native-like Phrasing)
    # Check 1: Base Corpus N-gram Fit (Collocation)
    collocation_score = 0.0
    base_cal = get_base_calibrator() if get_base_calibrator else None
    if base_cal and base_cal.is_available():
        base_fit = base_cal.score_doc(doc)
        if base_fit:
            # fit > 0.7 implies strong alignment with academic/native spoken corpus
            if base_fit.fit > 0.8:
                collocation_score = 9.0
            elif base_fit.fit > 0.7:
                collocation_score = 8.0
            elif base_fit.fit > 0.6:
                collocation_score = 7.0
            else:
                collocation_score = 5.0
            
            # Collocation boosts: only lift to next half-band, not jump
            if collocation_score >= 8.0 and lr_score >= 7.0:
                lr_score = min(lr_score + 0.5, 9.0)
                rationale.append(f"✅ **Collocation**: Strong native-like phrasing (Fit: {base_fit.fit:.2f}).")

    # Check 2: Colligation (Grammar-Lexis Interaction)
    # Look for specific dependency patterns that signal advanced usage
    # e.g., "Verb + Preposition" pairs that are correct, or "Adj + Noun" collocations
    colligation_hits = 0
    
    # Common verb-prep patterns (simplified check)
    verb_prep_patterns = [
        (r"\bdepend\s+on\b", "depend on"),
        (r"\brely\s+on\b", "rely on"),
        (r"\bconsist\s+of\b", "consist of"),
        (r"\bcontribute\s+to\b", "contribute to"),
        (r"\bsucceed\s+in\b", "succeed in"),
        (r"\bapologize\s+for\b", "apologize for"),
        (r"\bobject\s+to\b", "object to"),
        (r"\bapprove\s+of\b", "approve of"),
        (r"\bparticipate\s+in\b", "participate in"),
        (r"\bconcentrate\s+on\b", "concentrate on")
    ]
    
    for pattern, phrase in verb_prep_patterns:
        if re.search(pattern, text_lower):
            colligation_hits += 1
            
    # Colligation: modest boost, not a full band jump
    if colligation_hits >= 3 and lr_score < 7.0:
        lr_score += 0.5
        rationale.append("✅ **Colligation**: Correct verb-preposition usage detected.")

    scores['lexical'] = lr_score
    scores['collocation'] = collocation_score # Internal tracking

    if len(words) < 40 and scores['lexical'] > 7.0 and idiom_count < 1:
        scores['lexical'] = 7.0

    # 3. Grammatical Range & Accuracy (GRA)
    # RECALIBRATED to align with Official IELTS Band Descriptors
    # Band 9: "Structures precise and accurate at ALL TIMES"
    #         "Apart from 'mistakes' characteristic of native speaker speech"
    # Band 8: "Wide range flexibly used. MAJORITY of sentences error-free"
    #         "Occasional non-systematic errors"
    # Band 7: "Range of structures flexibly used. Error-free sentences FREQUENT"
    #         "Both simple and complex used effectively despite some errors"
    # Band 6: "Mix of short and complex sentence forms with LIMITED flexibility"
    #         "Errors frequently occur in complex structures"
    # Band 5: "Basic sentence forms fairly well controlled"
    #         "Complex structures attempted but LIMITED in range, nearly always contain errors"
    
    # Combine Regex and Dependency Parsing
    complex_types_found = set()
    # Regex checks (Keep existing logic as it's fast and specific)
    for s_type, patterns in COMPLEX_STRUCTURES.items():
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                complex_types_found.add(s_type)
                break
    
    # Add spaCy detected structures
    if has_passive: complex_types_found.add("Passive Voice")
    if has_relative: complex_types_found.add("Relative Clause")
    if has_subordinate: complex_types_found.add("Subordinate Clause")
    
    # Accuracy: Use LanguageTool
    error_count = 0
    gt = get_grammar_tool()
    if gt:
        matches = gt.check(text)
        error_count = len(matches)

    error_density = error_count / max(1, len(words))
    
    gra_score = 4.0
    
    # STRICTER: Dynamic Grammar Threshold
    # Require more complex structures per word count
    req_complex = max(2, int(len(words) / 18))  # Increased from /20, min 2 instead of 1
    req_complex = min(7, req_complex)  # Increased cap from 6
    
    # Part-specific adjustments
    if part == 3:
        req_complex = max(req_complex, 4)  # Increased from 3 - Part 3 requires more sophistication
    elif part == 1:
        req_complex = min(req_complex, 3)  # Cap at 3 for Part 1 (short answers)

    # STRICTER error density thresholds (Official descriptors are strict about accuracy)
    # Band 9: "precise and accurate at ALL times" = virtually no errors (< 1%)
    # Band 8: "MAJORITY error-free" = < 2% errors
    # Band 7: "Error-free sentences FREQUENT" = < 4% errors
    # Band 6: "Errors frequently occur" in complex structures = < 7% errors
    
    max_error_9 = 0.01   # Reduced from 0.03 - "accurate at ALL times"
    max_error_8 = 0.025  # Reduced from 0.05 - "majority error-free"
    max_error_7 = 0.04   # Reduced from 0.07 - "error-free sentences frequent"
    max_error_6 = 0.07   # Reduced from 0.10
    
    if len(complex_types_found) >= req_complex and error_density < max_error_9:
        gra_score = 9.0
        rationale.append("✅ **Grammar**: Band 9 - Structures are precise and accurate at all times.")
    elif len(complex_types_found) >= max(2, req_complex - 1) and error_density < max_error_8:
        gra_score = 8.0
        rationale.append("✅ **Grammar**: Band 8 - Wide range of structures; majority of sentences error-free.")
    elif len(complex_types_found) >= max(2, req_complex - 2) and error_density < max_error_7:
        gra_score = 7.0
        rationale.append("✅ **Grammar**: Band 7 - A range of structures; error-free sentences are frequent.")
    elif len(complex_types_found) >= 2 and error_density < max_error_6:
        gra_score = 6.0
        rationale.append("⚠️ **Grammar**: Band 6 - Mix of short and complex forms; errors in complex structures.")
    elif len(complex_types_found) >= 1 and error_density < 0.12:
        gra_score = 5.0
        rationale.append("⚠️ **Grammar**: Band 5 - Basic sentence forms fairly well controlled.")
    else:
        gra_score = 4.0
        rationale.append("❌ **Grammar**: Band 4 - Can produce basic sentence forms; subordinate clauses rare.")

    scores['grammar'] = gra_score
    scores['collocation'] = collocation_score # Internal tracking
    
    # 4. Pronunciation (PR)
    # RECALIBRATED to align with Official IELTS Band Descriptors
    # Band 9: "Full range of phonological features to convey precise/subtle meaning"
    #         "Can be EFFORTLESSLY understood throughout"
    #         "Accent has NO effect on intelligibility"
    # Band 8: "Wide range of phonological features to convey precise/subtle meaning"
    #         "Can sustain appropriate rhythm. Flexible stress and intonation"
    #         "Can be EASILY understood throughout. Accent has MINIMAL effect"
    # Band 7: "Uses range of features but control is VARIABLE"
    #         "Chunking generally appropriate, rhythm may be affected"
    #         "Can GENERALLY be understood without much effort"
    # Band 6: "Uses range of features but control is VARIABLE"
    #         "Some effective use of intonation/stress but NOT sustained"
    #         "Individual words MAY be mispronounced, occasional lack of clarity"
    # Band 5: "More complex speech causes disfluency, simpler may be fluent"
    
    # Pronunciation scoring calibrated to Whisper's exp(avg_logprob) distribution:
    #   Native-clear ≈ 0.78-0.86, good L2 ≈ 0.55-0.75, average L2 ≈ 0.40-0.55
    # Whisper confidence reflects intelligibility, not accent judgment.
    if pron_score > 0.80:
        pr_score = 9.0
        pr_detail = "Effortlessly understood; natural rhythm and intonation throughout."
    elif pron_score > 0.73:
        pr_score = 8.0
        pr_detail = "Easily understood; wide range of phonological features with only occasional lapses."
    elif pron_score > 0.65:
        pr_score = 7.0
        pr_detail = "Generally understood; shows control of stress and intonation but not always sustained."
    elif pron_score > 0.55:
        pr_score = 6.5
        pr_detail = "Can be followed without significant effort; some words may lack clarity."
    elif pron_score > 0.45:
        pr_score = 6.0
        pr_detail = "Variable control of pronunciation; occasional mispronunciations but meaning is clear."
    elif pron_score > 0.35:
        pr_score = 5.0
        pr_detail = "Noticeable pronunciation issues that sometimes affect intelligibility."
    elif pron_score > 0.25:
        pr_score = 4.0
        pr_detail = "Limited control of phonological features; listener effort is required."
    else:
        pr_score = 3.0
        pr_detail = "Frequent mispronunciations causing difficulty for the listener."

    conf_pct = int(pron_score * 100)
    pr_emoji = "✅" if pr_score >= 7.0 else ("⚠️" if pr_score >= 5.0 else "❌")
    rationale.append(f"{pr_emoji} **Pronunciation**: Band {pr_score} - {pr_detail} (ASR confidence: {conf_pct}%)")
        
    scores['pronunciation'] = pr_score

    base_cal = get_base_calibrator() if get_base_calibrator else None
    if base_cal and base_cal.is_available():
        base_fit = base_cal.score_doc(doc)
        if base_fit:
            if base_fit.boost > 0:
                scores['fluency'] = min(9.0, scores['fluency'] + base_fit.boost)
                scores['lexical'] = min(9.0, scores['lexical'] + base_fit.boost)
                markers = f" (Markers: {', '.join(base_fit.top_markers)})" if base_fit.top_markers else ""
                rationale.append(f"BASE: Academic spoken alignment {base_fit.fit:.2f} (Boost: +{base_fit.boost:.2f}){markers}")
            else:
                rationale.append(f"BASE: Academic spoken alignment {base_fit.fit:.2f}")

    # --- Semantic Communication Calibration (SANSee inspired) ---
    # Apply adaptive weighting to component scores based on semantic analysis
    # This ensures alignment between components and final score
    semantic_pred, alpha, semantic_rationale = _semantic_transfer_band(doc)
    
    if semantic_pred is not None and alpha > 0.05:
        # Cap semantic influence to prevent runaway score inflation
        alpha = min(0.25, alpha)
        if len(words) < 50:
            alpha = min(0.2, alpha * 1.1)
        
        # Only apply semantic calibration DOWNWARD or with modest upward cap (+0.5)
        current_avg = (scores['fluency'] + scores['lexical'] + scores['grammar'] + scores['pronunciation']) / 4.0
        if current_avg > 0:
            adjustment_factor = (semantic_pred - current_avg) * alpha
            # Cap upward adjustment to avoid inflation
            adjustment_factor = min(0.5, adjustment_factor)
            
            scores['fluency'] = max(1.0, min(9.0, scores['fluency'] + adjustment_factor * 0.8))
            scores['lexical'] = max(1.0, min(9.0, scores['lexical'] + adjustment_factor * 0.8))
            scores['grammar'] = max(1.0, min(9.0, scores['grammar'] + adjustment_factor * 0.6))
            scores['pronunciation'] = max(1.0, min(9.0, scores['pronunciation'] + adjustment_factor * 0.5))
            
            if alpha > 0.1:
                rationale.append(f"✅ **Semantic**: {semantic_rationale} (Weight: {alpha:.2f})")
    
    # --- CHAI Calibration (Polasa et al., 2025) ---
    # LLM evaluate() removed from scoring pipeline — too slow on CPU (3-5 min per call).
    # Rule-based + UniSep scores are sufficient. LLM is used only for model answer + grammar recs.
    try:
        from backend.chai_calibration import calibrate_score
        scores['fluency'] = calibrate_score(scores['fluency'], part, 'fluency')
        scores['lexical'] = calibrate_score(scores['lexical'], part, 'lexical')
        scores['grammar'] = calibrate_score(scores['grammar'], part, 'grammar')
        scores['pronunciation'] = calibrate_score(scores['pronunciation'], part, 'pronunciation')
        rationale.append(f"⚖️ **CHAI Calibration**: Applied human prior adjustment.")
    except ImportError:
        try:
            from chai_calibration import calibrate_score
            scores['fluency'] = calibrate_score(scores['fluency'], part, 'fluency')
            scores['lexical'] = calibrate_score(scores['lexical'], part, 'lexical')
            scores['grammar'] = calibrate_score(scores['grammar'], part, 'grammar')
            scores['pronunciation'] = calibrate_score(scores['pronunciation'], part, 'pronunciation')
            rationale.append(f"⚖️ **CHAI Calibration**: Applied human prior adjustment.")
        except ImportError:
            pass

            rationale.append(f"🤖 **AI Judge**: Hybrid scoring applied (Conf: {llm_conf:.2f}).")
            if "overall_rationale" in llm_result:
                rationale.append(f"💡 **Insight**: {llm_result['overall_rationale']}")

    # Round component scores to nearest 0.5 for display consistency
    scores['fluency'] = round(scores['fluency'] * 2) / 2
    scores['lexical'] = round(scores['lexical'] * 2) / 2
    scores['grammar'] = round(scores['grammar'] * 2) / 2
    scores['pronunciation'] = round(scores['pronunciation'] * 2) / 2
    scores['collocation'] = round(scores['collocation'], 1)
    
    # Final Calculation: ALWAYS derive from component scores for alignment
    # This ensures predicted score = average of component scores
    raw_score = (scores['fluency'] + scores['lexical'] + scores['grammar'] + scores['pronunciation']) / 4.0
    
    # Cap score if grammar errors are excessive (automatic failsafe)
    if error_density > 0.15:
        raw_score = min(6.0, raw_score)
        # Also cap grammar component for consistency
        scores['grammar'] = min(5.0, scores['grammar'])
        raw_score = (scores['fluency'] + scores['lexical'] + scores['grammar'] + scores['pronunciation']) / 4.0
    
    # Final score = rounded average of rounded components
    final_score = round(raw_score * 2) / 2
    final_score = max(1.0, final_score)
    
    # Sanity check: Ensure final score is consistent with components
    expected_final = (scores['fluency'] + scores['lexical'] + scores['grammar'] + scores['pronunciation']) / 4.0
    expected_final_rounded = round(expected_final * 2) / 2
    if abs(final_score - expected_final_rounded) > 0.1:
        # Recalibrate to ensure alignment
        final_score = expected_final_rounded
    
    # Update rationale to show FINAL calibrated scores (ensures Score Analysis matches breakdown display)
    # Remove old band rationale entries and add updated ones with final scores
    updated_rationale = []
    for r in rationale:
        # Skip old band entries that will be replaced with final scores
        if r.startswith("✅ **Fluency**:") or r.startswith("⚠️ **Fluency**:") or r.startswith("❌ **Fluency**:"):
            continue
        if r.startswith("✅ **Lexical**:") or r.startswith("⚠️ **Lexical**:") or r.startswith("❌ **Lexical**:"):
            continue
        if r.startswith("✅ **Grammar**:") or r.startswith("⚠️ **Grammar**:") or r.startswith("❌ **Grammar**:"):
            continue
        if r.startswith("✅ **Pronunciation**:") or r.startswith("⚠️ **Pronunciation**:") or r.startswith("❌ **Pronunciation**:"):
            continue
        updated_rationale.append(r)
    
    # Add final calibrated scores to rationale (these will match the breakdown display exactly)
    def _band_emoji(score):
        if score >= 7.0:
            return "✅"
        elif score >= 5.0:
            return "⚠️"
        else:
            return "❌"
    
    def _band_desc(criterion, score):
        # Official IELTS Band Descriptors - aligned with stricter calibration
        descs = {
            "Fluency": {
                9.0: "Fluent; hesitation used only to prepare content.",
                8.5: "Fluent; hesitation mostly content-related, occasionally for words.",
                8.0: "Fluent; hesitation mostly content-related, occasionally for words.",
                7.5: "Able to keep going and readily produce long turns without noticeable effort.",
                7.0: "Able to keep going and readily produce long turns without noticeable effort.",
                6.5: "Willing to produce long turns but loses coherence at times.",
                6.0: "Willing to produce long turns but loses coherence at times.",
                5.5: "Usually able to keep going, but relies on repetition and self-correction.",
                5.0: "Usually able to keep going, but relies on repetition and self-correction.",
                4.5: "Unable to keep going without noticeable pauses.",
                4.0: "Unable to keep going without noticeable pauses.",
            },
            "Lexical": {
                9.0: "Total flexibility and precise use in all contexts.",
                8.5: "Wide resource, readily and flexibly used.",
                8.0: "Wide resource, readily and flexibly used.",
                7.5: "Resource flexibly used to discuss a variety of topics.",
                7.0: "Resource flexibly used to discuss a variety of topics.",
                6.5: "Resource sufficient to discuss topics at length.",
                6.0: "Resource sufficient to discuss topics at length.",
                5.5: "Resource sufficient for familiar topics but limited flexibility.",
                5.0: "Resource sufficient for familiar topics but limited flexibility.",
                4.5: "Resource sufficient for familiar topics but basic meaning only.",
                4.0: "Resource sufficient for familiar topics but basic meaning only.",
            },
            "Grammar": {
                9.0: "Structures are precise and accurate at all times.",
                8.5: "Wide range of structures, flexibly used; majority error-free.",
                8.0: "Wide range of structures, flexibly used; majority error-free.",
                7.5: "A range of structures flexibly used; error-free sentences frequent.",
                7.0: "A range of structures flexibly used; error-free sentences frequent.",
                6.5: "Mix of short and complex sentence forms.",
                6.0: "Mix of short and complex sentence forms.",
                5.5: "Basic sentence forms are fairly well controlled.",
                5.0: "Basic sentence forms are fairly well controlled.",
                4.5: "Can produce basic sentence forms and some short utterances are error-free.",
                4.0: "Can produce basic sentence forms and some short utterances are error-free.",
            },
            "Pronunciation": {
                9.0: "Full range of phonological features; effortlessly understood.",
                8.5: "Wide range of phonological features; easily understood.",
                8.0: "Wide range of phonological features; easily understood.",
                7.5: "Shows all positive features of Band 6 and some of Band 8.",
                7.0: "Shows all positive features of Band 6 and some of Band 8.",
                6.5: "Range of phonological features, but control is variable.",
                6.0: "Range of phonological features, but control is variable.",
                5.5: "Shows all positive features of Band 4 and some of Band 6.",
                5.0: "Shows all positive features of Band 4 and some of Band 6.",
                4.5: "Uses some acceptable phonological features, but range is limited.",
                4.0: "Uses some acceptable phonological features, but range is limited.",
                3.0: "Difficult to understand.",
            }
        }
        # Find closest band description
        criterion_descs = descs.get(criterion, {})
        closest = min(criterion_descs.keys(), key=lambda x: abs(x - score), default=score)
        return criterion_descs.get(closest, f"Band {score}")
    
    updated_rationale.insert(0, f"{_band_emoji(scores['pronunciation'])} **Pronunciation**: Band {scores['pronunciation']} - {_band_desc('Pronunciation', scores['pronunciation'])}")
    updated_rationale.insert(0, f"{_band_emoji(scores['grammar'])} **Grammar**: Band {scores['grammar']} - {_band_desc('Grammar', scores['grammar'])}")
    updated_rationale.insert(0, f"{_band_emoji(scores['lexical'])} **Lexical**: Band {scores['lexical']} - {_band_desc('Lexical', scores['lexical'])}")
    updated_rationale.insert(0, f"{_band_emoji(scores['fluency'])} **Fluency**: Band {scores['fluency']} - {_band_desc('Fluency', scores['fluency'])}")
    
    return final_score, scores, updated_rationale

import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

# --- Deep Learning Model Integration ---
try:
    from backend.models.scoring_model import load_model
    from backend.features import extract_features
    # Load model on startup (or lazy load)
    # Since we don't have weights yet, this will initialize random weights.
    # In production, path would be "resources/scoring_model.pth"
    dl_scoring_model = load_model()
    print("Deep Learning Scoring Model initialized.")
except ImportError as e:
    print(f"DL Model Import Error: {e}")
    dl_scoring_model = None
    extract_features = None

try:
    from backend.telemetry import log_scoring_event
except Exception:
    from telemetry import log_scoring_event

# --- UniSep-Inspired Pronunciation Scorer (EnCodec + Transformer) ---
unisep_model = None
unisep_tokenizer = None
try:
    try:
        from backend.models.unisep_scorer import load_unisep_model, score_audio as unisep_score_audio
    except ImportError:
        from models.unisep_scorer import load_unisep_model, score_audio as unisep_score_audio
    unisep_model, unisep_tokenizer = load_unisep_model()
except ImportError as e:
    print(f"UniSep Import Error: {e}")
except Exception as e:
    print(f"UniSep Init Error: {e}")

# Initialize ThreadPoolExecutor for heavy tasks
# 3 workers: model_answer + grammar_recs can run in parallel + ASR/scoring
executor = ThreadPoolExecutor(max_workers=3)

import gc

# In-memory task store (Use Redis/DB in production)
# Format: { task_id: { "status": "pending"|"processing"|"completed"|"failed", "result": {...}, "error": str, "created_at": timestamp } }
tasks: Dict[str, Dict] = {}

def _debug_log(*args, **kwargs):
    pass

def _agent_debug_log(*args, **kwargs):
    pass

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

async def process_audio_task(task_id: str, file_content: bytes, filename: str, part: int, user_id: str = "default"):
    """
    Background task to process audio analysis.
    """
    # #region agent log
    _agent_debug_log("H3", "main.py:process_audio_task:start", "background task start", {
        "task_id": task_id,
        "exists_before_start": task_id in tasks,
        "tasks_size": len(tasks),
        "filename": filename,
        "part": part,
        "user_id": user_id,
    }, run_id="pre-fix-2")
    # #endregion
    tasks[task_id]["status"] = "processing"
    logger.info(f"Task {task_id}: starting (file={filename}, part={part}, user_id={user_id}, size={len(file_content)} bytes)")
    
    # Create temp file from memory
    suffix = f".{filename.split('.')[-1]}" if '.' in filename else ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    processed_path = None
    
    try:
        with open(tmp_path, "wb") as buffer:
            buffer.write(file_content)
            
        # 1. Digital Signal Processing (Run in ThreadPool)
        logger.info(f"Task {task_id}: [1/5] DSP starting...")
        loop = asyncio.get_event_loop()
        processed_path, dsp_duration = await loop.run_in_executor(
            executor, process_audio_signal, tmp_path
        )
        logger.info(f"Task {task_id}: [1/5] DSP done (duration={dsp_duration:.1f}s)")
        
        temporal = compute_temporal_fluency_features(processed_path)
        mfcc_summary = compute_mfcc_summary(processed_path)
        
        if dsp_duration == 0.0:
            dsp_duration = 1.0 
        
        # --- Deep Learning Model Inference (Run in Parallel) ---
        dl_scores = None
        
        def run_dl_scoring_sync():
            if dl_scoring_model and extract_features:
                try:
                    mfcc, prosody, static = extract_features(processed_path)
                    if mfcc is not None:
                        dl_scoring_model.eval()
                        with torch.no_grad():
                            preds = dl_scoring_model(mfcc, prosody, static)
                            return {k: max(1.0, min(9.0, float(v.item()))) for k, v in preds.items()}
                except Exception as e:
                    logger.warning(f"Task {task_id}: DL Scoring Failed: {e}")
            return None

        # Launch DL scoring task
        dl_task = loop.run_in_executor(executor, run_dl_scoring_sync)

        # --- UniSep-Inspired Pronunciation Scoring (runs in parallel) ---
        def run_unisep_scoring_sync():
            if unisep_model and unisep_tokenizer:
                try:
                    y, _ = librosa.load(processed_path, sr=16000, duration=60)
                    return unisep_score_audio(unisep_model, unisep_tokenizer, y)
                except Exception as e:
                    logger.warning(f"Task {task_id}: UniSep scoring failed: {e}")
            return None

        unisep_task = loop.run_in_executor(executor, run_unisep_scoring_sync)

        # 2. ASR Transcription via Subprocess
        logger.info(f"Task {task_id}: [2/5] ASR transcription starting...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcribe.py")
        ASR_TIMEOUT = 300  # 5 minutes max per model attempt
        
        def run_transcription():
            # "base" first for acceptable speed on CPU; fall back to "tiny"
            models_to_try = ["tiny", "base"]  # tiny first for speed on CPU
            env = os.environ.copy()
            result = None
            # #region agent log
            _debug_log(
                "M0",
                "backend/main.py:run_transcription:entry",
                "Run transcription entry",
                {
                    "task_id": task_id,
                    "script_path": script_path,
                    "processed_path": processed_path,
                    "processed_exists": os.path.exists(processed_path) if processed_path else False,
                    "python_executable": sys.executable,
                    "models_to_try": models_to_try,
                },
            )
            # #endregion
            
            for model_name in models_to_try:
                logger.info(f"Task {task_id}: ASR trying model '{model_name}'...")
                try:
                    # #region agent log
                    _debug_log(
                        "M1",
                        "backend/main.py:run_transcription:before_subprocess",
                        "Before ASR subprocess.run",
                        {"task_id": task_id, "model_name": model_name},
                    )
                    # #endregion
                    result = subprocess.run(
                        [sys.executable, script_path, processed_path, model_name],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=ASR_TIMEOUT
                    )
                    # #region agent log
                    _debug_log(
                        "M2",
                        "backend/main.py:run_transcription:after_subprocess",
                        "After ASR subprocess.run",
                        {
                            "task_id": task_id,
                            "model_name": model_name,
                            "returncode": result.returncode,
                            "stdout_head": (result.stdout or "")[:180],
                            "stderr_head": (result.stderr or "")[:180],
                        },
                    )
                    # #endregion
                except subprocess.TimeoutExpired:
                    logger.warning(f"Task {task_id}: model '{model_name}' timed out after {ASR_TIMEOUT}s")
                    # #region agent log
                    _debug_log(
                        "M3",
                        "backend/main.py:run_transcription:timeout",
                        "ASR subprocess timed out",
                        {"task_id": task_id, "model_name": model_name, "timeout_s": ASR_TIMEOUT},
                    )
                    # #endregion
                    continue
                
                if result.returncode == 0:
                    try:
                        json.loads(result.stdout)
                        logger.info(f"Task {task_id}: ASR succeeded with model '{model_name}'")
                        return result
                    except json.JSONDecodeError:
                        logger.warning(f"Task {task_id}: model '{model_name}' returned invalid JSON")
                        continue
                else:
                    logger.warning(f"Task {task_id}: model '{model_name}' failed: {result.stderr[:200]}")
                    
            return result
            
        process = await loop.run_in_executor(executor, run_transcription)
        
        if process is None or process.returncode != 0:
            stderr_msg = process.stderr[:500] if process else "all models timed out"
            # #region agent log
            _debug_log(
                "M4",
                "backend/main.py:process_audio_task:asr_failed",
                "ASR failed before parse",
                {
                    "task_id": task_id,
                    "process_is_none": process is None,
                    "returncode": None if process is None else process.returncode,
                    "stderr_head": stderr_msg[:300],
                },
            )
            # #endregion
            raise Exception(f"ASR Failed: {stderr_msg}")

        try:
            asr_result = json.loads(process.stdout)
            text = asr_result.get("text", "")
            asr_model = asr_result.get("model", "unknown")
            asr_avg_logprob = asr_result.get("avg_logprob", -1.0)
            pron_score = asr_result.get("pron_score", 0.5)
        except json.JSONDecodeError:
            raise Exception("Invalid ASR Output")

        if not text:
             raise Exception("No speech detected")

        logger.info(f"Task {task_id}: [2/5] ASR done (model={asr_model}, words={len(text.split())}, pron_score={pron_score:.3f})")

        # 3. Text Analysis & Scoring (Run in ThreadPool)
        if mfcc_summary:
            if temporal is None: temporal = {}
            temporal.update(mfcc_summary)

        def analysis_task():
            recs = get_smart_recommendations(text, part, user_id=user_id)
            sc, bd, rat = calculate_score_with_rationale(text, dsp_duration, pron_score, part, temporal)
            return recs, sc, bd, rat
        
        logger.info(f"Task {task_id}: [3/5] Text analysis starting...")
        recommendations, score, breakdown, rationale = await loop.run_in_executor(executor, analysis_task)
        logger.info(f"Task {task_id}: [3/5] Text analysis done (score={score})")
        
        # Grammar recs from LanguageTool (fast; skip LLM grammar to save ~5 min)
        grammar_recs = []
        gt = get_grammar_tool()
        if gt:
            for match in gt.check(text)[:3]:
                original = _extract_languagetool_original(text, match)
                repls = _extract_languagetool_replacements(match)
                suggestion = repls[0] if repls else "Check grammar"
                msg = _extract_languagetool_message(match)
                rid = _extract_languagetool_rule_id(match)
                grammar_recs.append({
                    "original": original,
                    "suggestion": suggestion,
                    "explanation": f"{msg} (Rule: {rid})" if rid else msg,
                    "type": "Correction"
                })
        if not grammar_recs:
            grammar_recs.append({
                "original": "Overall response",
                "suggestion": "Try using more complex sentence structures.",
                "explanation": "No major grammatical errors detected. To reach Band 7+, aim to use a wider range of subordinate clauses and conditional sentences.",
                "type": "Enhancement"
            })
        
        # Grammar penalty based on detected errors
        correction_count = sum(1 for r in grammar_recs if r.get('type') == 'Correction')
        if correction_count > 0:
            current_gra = breakdown.get('grammar', 6.0)
            new_gra = current_gra
            if correction_count >= 5:
                new_gra = min(current_gra, 6.0)
            elif correction_count >= 3:
                new_gra = min(current_gra, 7.0)
            elif correction_count >= 1:
                new_gra = min(current_gra, 8.0)
            if new_gra < current_gra:
                breakdown['grammar'] = float(new_gra)
                rationale.append(f"📉 **Grammar Penalty**: Adjusted to Band {new_gra} due to {correction_count} detected errors.")
                f = breakdown.get('fluency', 0)
                l = breakdown.get('lexical', 0)
                p = breakdown.get('pronunciation', 0)
                avg = (f + l + new_gra + p) / 4
                decimal = avg - int(avg)
                if decimal >= 0.75:
                    score = float(int(avg) + 1.0)
                elif decimal >= 0.25:
                    score = float(int(avg) + 0.5)
                else:
                    score = float(int(avg))
        
        # Model Answer targeting candidate's score + 1 band (i+1 principle)
        model_answer = None
        user_profile = get_user_profile(user_id)
        learner_state = get_learner_state(user_id) if ONTOLOGY_TRAJECTORY_ENABLED else None
        trajectory_context = {}
        if learner_state is not None and learner_state.trajectory_plan:
            idx = learner_state.trajectory_step
            if 0 <= idx < len(learner_state.trajectory_plan):
                trajectory_context["trajectory_step"] = learner_state.trajectory_plan[idx]
            trajectory_context["target_concepts"] = [
                s.get("concept")
                for s in learner_state.trajectory_plan[:3]
                if isinstance(s, dict)
            ]
            trajectory_context["register_goal"] = (
                getattr(user_profile, "register_preference", None) or "neutral"
            )
        evaluator = get_llm_evaluator()
        if evaluator and evaluator.is_available():
            logger.info(f"Task {task_id}: [4/5] Model answer generation starting (user_band={score})...")
            LLM_TIMEOUT = 300
            try:
                model_answer = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        evaluator.generate_model_answer,
                        text,
                        f"Part {part}",
                        score,
                        trajectory_context,
                    ),
                    timeout=LLM_TIMEOUT,
                )
                logger.info(f"Task {task_id}: [4/5] Model answer done ({len(model_answer) if model_answer else 0} chars)")
            except asyncio.TimeoutError:
                model_answer = "Model answer generation timed out (LLM took >5 min). Try a shorter response."
                logger.warning(f"Task {task_id}: [4/5] Model answer timed out")
            except Exception as e:
                model_answer = f"Model answer generation failed: {e}. Please try again."
                logger.exception(f"Task {task_id}: [4/5] Model answer error")
        
        # Fallback when LLM unavailable, model_answer empty, or LLM returned error message
        err_phrases = ("unavailable", "failed", "timed out", "too short")
        is_error = not model_answer or not str(model_answer).strip()
        if not is_error:
            ma_lower = str(model_answer).lower()
            is_error = any(p in ma_lower for p in err_phrases)
        if is_error:
            model_answer = _fallback_model_answer(text, part, user_band=score)
            logger.info(f"Task {task_id}: Using fallback model answer ({len(model_answer)} chars)")

        # --- Collect Deep Learning Model Results ---
        logger.info(f"Task {task_id}: [5/5] Collecting DL + UniSep scores...")
        try:
            dl_scores = await dl_task
            if dl_scores:
                breakdown['ai_beta_scores'] = dl_scores
                rationale.append(f"🧪 **Deep Learning Model**: Experimental Prediction Band {dl_scores['overall']:.1f}")
        except Exception as e:
            logger.warning(f"Task {task_id}: DL scores failed: {e}")

        # --- Collect UniSep Pronunciation Scores ---
        unisep_scores = None
        try:
            unisep_scores = await unisep_task
            if unisep_scores:
                # Blend UniSep with Whisper-based pronunciation
                # UniSep trained on overall score (not pronunciation-specific), so weight conservatively
                whisper_pron = breakdown.get('pronunciation', 6.0)
                unisep_pron = unisep_scores['pronunciation']
                blended_pron = 0.5 * unisep_pron + 0.5 * whisper_pron
                blended_pron = round(blended_pron * 2) / 2  # IELTS half-band rounding

                old_pron = breakdown['pronunciation']
                breakdown['pronunciation'] = blended_pron
                breakdown['unisep_scores'] = unisep_scores

                # Recalculate overall score with updated pronunciation
                f = breakdown.get('fluency', 0)
                l = breakdown.get('lexical', 0)
                g = breakdown.get('grammar', 0)
                raw = (f + l + g + blended_pron) / 4.0
                score = round(raw * 2) / 2
                score = max(1.0, score)

                rationale.append(
                    f"🎯 **UniSep Pronunciation Model**: Band {unisep_pron:.1f} "
                    f"(blended {old_pron}→{blended_pron} with codec-based scoring)"
                )
                logger.info(f"Task {task_id}: UniSep pron={unisep_pron:.1f}, blended={blended_pron}")
        except Exception as e:
            logger.warning(f"Task {task_id}: UniSep scores failed: {e}")

        response = {
            "request_id": task_id,
            "user_id": user_id,
            "asr_model": asr_model,
            "asr_avg_logprob": asr_avg_logprob,
            "mfcc_summary": mfcc_summary,
            "score": score,
            "breakdown": breakdown,
            "transcription": text,
            "rationale": rationale,
            "recommendations": recommendations,
            "grammar_recommendations": grammar_recs,
            "part": part,
            "temporal": temporal,
            "word_count": len(text.split()),
            "duration": round(dsp_duration, 2),
            "model_answer": model_answer
        }
        update_session_remap(user_id, text, part, breakdown, recommendations)
        try:
            log_scoring_event(response, request_id=task_id, mode="audio")
        except Exception:
            pass
        
        # Update Task Status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = response
        logger.info(f"Task {task_id}: COMPLETED (score={score})")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        
    finally:
        # Cleanup temp files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if processed_path and os.path.exists(processed_path) and processed_path != tmp_path:
            os.unlink(processed_path)

@app.post("/analyze_audio")
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    part: int = Form(1),
    user_id: str = Form("default"),
):
    """
    Async endpoint that accepts audio upload and starts background processing.
    Returns task_id immediately.
    If clients see ECONNRESET: ensure backend is reachable, same network, no proxy
    timeouts (e.g. nginx proxy_read_timeout / client_max_body_size), and stable WiFi.
    """
    task_id = str(uuid.uuid4())
    # Read file content into memory to pass to background task
    # Note: For very large files, streaming to disk first is better; audio is usually <10MB
    try:
        file_content = await file.read()
    except Exception as e:
        logger.exception("analyze_audio: failed to read upload body")
        raise HTTPException(status_code=400, detail=f"Upload read failed: {e!s}")
    
    # Initialize task status
    tasks[task_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "user_id": user_id,
        "result": None,
        "error": None
    }
    # #region agent log
    _agent_debug_log("H1", "main.py:analyze_audio:create", "task created", {
        "task_id": task_id,
        "tasks_size": len(tasks),
        "filename": file.filename,
        "part": part,
        "user_id": user_id,
    }, run_id="pre-fix-2")
    # #endregion
    
    # Start background task
    background_tasks.add_task(process_audio_task, task_id, file_content, file.filename, part, user_id)
    
    return {"task_id": task_id, "status": "pending", "user_id": user_id, "message": "Analysis started in background"}

@app.get("/task_status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Poll this endpoint to get the status and result of an analysis task.
    """
    # #region agent log
    _agent_debug_log("H2", "main.py:get_task_status:entry", "task status polled", {
        "task_id": task_id,
        "exists": task_id in tasks,
        "tasks_size": len(tasks),
        "sample_task_ids": list(tasks.keys())[:5],
    }, run_id="pre-fix-2")
    # #endregion
    if task_id not in tasks:
        return TaskStatusResponse(
            task_id=task_id,
            status="failed",
            error="Task expired or server restarted. Please record again."
        )
    
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )
    
    
# Deprecated synchronous endpoint (renamed/kept for reference if needed, but replaced by async version above)
# The original logic is now inside process_audio_task
@app.post("/analyze_audio_sync")
async def analyze_audio_sync(file: UploadFile = File(...), part: int = Form(1), user_id: str = Form("default")):
    request_id = str(uuid.uuid4())
    # Save uploaded file
    suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    processed_path = None
    
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Digital Signal Processing (Run in ThreadPool)
        loop = asyncio.get_event_loop()
        processed_path, dsp_duration = await loop.run_in_executor(
            executor, process_audio_signal, tmp_path
        )
        
        temporal = compute_temporal_fluency_features(processed_path)
        mfcc_summary = compute_mfcc_summary(processed_path)
        
        if dsp_duration == 0.0:
            dsp_duration = 1.0 
        
        # --- Deep Learning Model Inference (Run in Parallel) ---
        dl_scores = None
        
        def run_dl_scoring_sync():
            if dl_scoring_model and extract_features:
                try:
                    # Extract multi-modal features
                    mfcc, prosody, static = extract_features(processed_path)
                    if mfcc is not None:
                        # Forward pass
                        dl_scoring_model.eval()
                        with torch.no_grad():
                            preds = dl_scoring_model(mfcc, prosody, static)
                            # Clamp predictions to Band 1.0 - 9.0
                            return {k: max(1.0, min(9.0, float(v.item()))) for k, v in preds.items()}
                except Exception as e:
                    print(f"DL Scoring Failed: {e}")
            return None

        # Launch DL scoring task
        dl_task = loop.run_in_executor(executor, run_dl_scoring_sync)

        # 2. ASR Transcription via Subprocess (Stable & Robust)
        # Using "base" model for better accuracy
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcribe.py")
        
        def run_transcription():
            # Prioritize accuracy: Try 'small' -> 'base' -> 'tiny'
            # 'small' offers a significant jump in accuracy over 'base' for non-native speech
            
            models_to_try = ["small", "base", "tiny"]
            env = os.environ.copy()
            
            for model_name in models_to_try:
                print(f"Attempting transcription with model: {model_name}...")
                result = subprocess.run(
                    [sys.executable, script_path, processed_path, model_name],
                    capture_output=True,
                    text=True,
                    env=env
                )
                
                if result.returncode == 0:
                    # Validate JSON output
                    try:
                        json.loads(result.stdout)
                        return result # Success
                    except json.JSONDecodeError:
                        print(f"Model {model_name} returned invalid JSON. Retrying...")
                        continue
                else:
                    print(f"Model {model_name} failed with error: {result.stderr}")
                    
            # If all fail, return the last result (likely tiny failed)
            return result
        
        proc_result = await loop.run_in_executor(executor, run_transcription)
        
        if proc_result.returncode != 0:
             return {"error": f"Transcription failed: {proc_result.stderr}"}
             
        try:
            output = json.loads(proc_result.stdout)
        except json.JSONDecodeError:
             return {"error": f"Invalid output from transcriber: {proc_result.stdout}"}
             
        if "error" in output:
             return {"error": output["error"]}
             
        text = output.get("text", "")
        pron_score = output.get("pron_score", 0.5)
        asr_model = output.get("asr_model", None)
        asr_avg_logprob = output.get("avg_logprob", None)
        
        if not text:
             return {"error": "No speech detected in audio."}

        # 4. Semantic Analysis (Run in ThreadPool if heavy)
        # Note: calculate_score_with_rationale now uses spaCy which is CPU intensive
        if temporal:
            speech_dur_s = float(temporal.get("speech_dur_s", 0.0))
            if speech_dur_s > 0:
                temporal["articulation_wpm"] = (len(text.split()) / speech_dur_s) * 60
        
        # 5. Auditory Feature Extraction (MFCC/Gammatone) - "High-Pitch Voice Details"
        if mfcc_summary:
            if temporal is None: temporal = {}
            temporal.update(mfcc_summary)

        def analysis_task():
            recs = get_smart_recommendations(text, part, user_id=user_id)
            sc, bd, rat = calculate_score_with_rationale(text, dsp_duration, pron_score, part, temporal)
            return recs, sc, bd, rat
            
        recommendations, score, breakdown, rationale = await loop.run_in_executor(executor, analysis_task)
        
        # Retrieve user profile for adaptive feedback
        user_profile = get_user_profile(user_id)

        # Generate Model Answer at candidate's score + 1 band & Grammar Recommendations
        model_answer = None
        grammar_recs = []
        
        evaluator = get_llm_evaluator()
        if evaluator and evaluator.is_available():
            learner_state = get_learner_state(user_id) if ONTOLOGY_TRAJECTORY_ENABLED else None
            trajectory_context = {}
            if learner_state is not None and learner_state.trajectory_plan:
                idx = learner_state.trajectory_step
                if 0 <= idx < len(learner_state.trajectory_plan):
                    trajectory_context["trajectory_step"] = learner_state.trajectory_plan[idx]
                trajectory_context["target_concepts"] = [
                    s.get("concept")
                    for s in learner_state.trajectory_plan[:3]
                    if isinstance(s, dict)
                ]
                trajectory_context["register_goal"] = user_profile.register_preference or "neutral"
            model_answer = await loop.run_in_executor(
                executor,
                evaluator.generate_model_answer,
                text,
                f"Part {part}",
                score,
                trajectory_context,
            )
            
            # Generate adaptive grammar feedback based on ZPD
            grammar_recs = await loop.run_in_executor(
                executor, 
                evaluator.generate_grammar_recommendations, 
                text, 
                user_profile.proficiency_level
            )
            
            # Post-Processing: Penalize Grammar Score based on LLM-detected errors
            correction_count = sum(1 for r in grammar_recs if r.get('type') == 'Correction')
            if correction_count > 0:
                current_gra = breakdown.get('grammar', 6.0)
                
                # Apply penalty based on error count
                new_gra = current_gra
                if correction_count >= 5:
                    new_gra = min(current_gra, 6.0)
                elif correction_count >= 3:
                    new_gra = min(current_gra, 7.0)
                elif correction_count >= 1:
                    new_gra = min(current_gra, 8.0)
                    
                if new_gra < current_gra:
                    breakdown['grammar'] = float(new_gra)
                    rationale.append(f"📉 **Grammar Penalty**: Adjusted to Band {new_gra} due to {correction_count} detected errors.")
                    
                    # Recalculate Overall Score
                    f = breakdown.get('fluency', 0)
                    l = breakdown.get('lexical', 0)
                    p = breakdown.get('pronunciation', 0)
                    avg = (f + l + new_gra + p) / 4
                    
                    # IELTS Rounding
                    decimal = avg - int(avg)
                    if decimal >= 0.75:
                        score = float(int(avg) + 1.0)
                    elif decimal >= 0.25:
                        score = float(int(avg) + 0.5)
                    else:
                        score = float(int(avg))
            
        # Fallback: Use LanguageTool if LLM unavailable or returned no recs
        if not grammar_recs:
            gt = get_grammar_tool()
            if gt:
                matches = gt.check(text)
                # Take top 3 errors
                for match in matches[:3]:
                    original = _extract_languagetool_original(text, match)
                    repls = _extract_languagetool_replacements(match)
                    suggestion = repls[0] if repls else "Check grammar"
                    msg = _extract_languagetool_message(match)
                    rid = _extract_languagetool_rule_id(match)
                    grammar_recs.append({
                    "original": original,
                    "suggestion": suggestion,
                    "explanation": f"{msg} (Rule: {rid})" if rid else msg,
                    "type": "Correction"
                })
        
        # Fallback 2: If still empty (no errors found), add positive reinforcement
        if not grammar_recs:
             grammar_recs.append({
                "original": "Overall response",
                "suggestion": "Try using more complex sentence structures.",
                "explanation": "No major grammatical errors detected. To reach Band 7+, aim to use a wider range of subordinate clauses and conditional sentences.",
                "type": "Enhancement"
            })

        # Fallback model answer when LLM unavailable or returned error
        err_phrases = ("unavailable", "failed", "timed out", "too short")
        is_error = not model_answer or not str(model_answer).strip()
        if not is_error:
            ma_lower = str(model_answer).lower()
            is_error = any(p in ma_lower for p in err_phrases)
        if is_error:
            model_answer = _fallback_model_answer(text, part, user_band=score)

        # --- Collect Deep Learning Model Results ---
        try:
            dl_scores = await dl_task
            if dl_scores:
                # Add to breakdown as experimental feature
                breakdown['ai_beta_scores'] = dl_scores
                rationale.append(f"🧪 **Deep Learning Model**: Experimental Prediction Band {dl_scores['overall']:.1f}")
        except Exception as e:
            print(f"Failed to retrieve DL scores: {e}")

        response = {
            "request_id": request_id,
            "user_id": user_id,
            "asr_model": asr_model,
            "asr_avg_logprob": asr_avg_logprob,
            "mfcc_summary": mfcc_summary,
            "score": score,
            "breakdown": breakdown,
            "transcription": text,
            "rationale": rationale,
            "recommendations": recommendations,
            "grammar_recommendations": grammar_recs,
            "part": part,
            "temporal": temporal,
            "word_count": len(text.split()),
            "duration": round(dsp_duration, 2),
            "model_answer": model_answer
        }
        update_session_remap(user_id, text, part, breakdown, recommendations)
        try:
            log_scoring_event(response, request_id=request_id, mode="audio")
        except Exception:
            pass
        return response
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Cleanup temp files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if processed_path and os.path.exists(processed_path) and processed_path != tmp_path:
            os.unlink(processed_path)

@app.post("/analyze")
async def analyze_text(input: TextInput):
    request_id = str(uuid.uuid4())
    text = input.text
    duration = input.duration
    pron_score = input.pronunciation_score
    part = input.part
    user_id = (input.user_id or "default").strip() or "default"
    
    recommendations = get_smart_recommendations(text, part, user_id=user_id)
    score, breakdown, rationale = calculate_score_with_rationale(text, duration, pron_score, part)
    
    # Retrieve user profile
    user_profile = get_user_profile(user_id)
    
    # Generate Model Answer at candidate's score + 1 band & Grammar Recs
    model_answer = None
    grammar_recs = []
    
    evaluator = get_llm_evaluator()
    if evaluator and evaluator.is_available():
        learner_state = get_learner_state(user_id) if ONTOLOGY_TRAJECTORY_ENABLED else None
        trajectory_context = {}
        if learner_state is not None and learner_state.trajectory_plan:
            idx = learner_state.trajectory_step
            if 0 <= idx < len(learner_state.trajectory_plan):
                trajectory_context["trajectory_step"] = learner_state.trajectory_plan[idx]
            trajectory_context["target_concepts"] = [
                s.get("concept")
                for s in learner_state.trajectory_plan[:3]
                if isinstance(s, dict)
            ]
            trajectory_context["register_goal"] = user_profile.register_preference or "neutral"
        model_answer = evaluator.generate_model_answer(text, user_band=score, trajectory_context=trajectory_context)
        grammar_recs = evaluator.generate_grammar_recommendations(text, user_profile.proficiency_level)

    # Fallback: Use LanguageTool if LLM unavailable or returned no recs
    if not grammar_recs:
        gt = get_grammar_tool()
        if gt:
            matches = gt.check(text)
            for match in matches[:3]:
                original = _extract_languagetool_original(text, match)
                repls = _extract_languagetool_replacements(match)
                suggestion = repls[0] if repls else "Check grammar"
                msg = _extract_languagetool_message(match)
                rid = _extract_languagetool_rule_id(match)
                grammar_recs.append({
                    "original": original,
                    "suggestion": suggestion,
                    "explanation": f"{msg} (Rule: {rid})" if rid else msg,
                    "type": "Correction"
                })

    if not grammar_recs:
        grammar_recs.append({
            "original": "Overall response",
            "suggestion": "Try using more complex sentence structures.",
            "explanation": "No major grammatical errors detected. To reach Band 7+, aim to use a wider range of subordinate clauses and conditional sentences.",
            "type": "Enhancement"
        })

    # Fallback model answer when LLM unavailable or returned error
    err_phrases = ("unavailable", "failed", "timed out", "too short")
    is_error = not model_answer or not str(model_answer).strip()
    if not is_error:
        ma_lower = str(model_answer).lower()
        is_error = any(p in ma_lower for p in err_phrases)
    if is_error:
        model_answer = _fallback_model_answer(text, part, user_band=score)

    response = {
        "request_id": request_id,
        "user_id": user_id,
        "score": score,
        "breakdown": breakdown,
        "recommendations": recommendations,
        "grammar_recommendations": grammar_recs,
        "rationale": rationale,
        "word_count": len(text.split()),
        "part": part,
        "model_answer": model_answer
    }
    update_session_remap(user_id, text, part, breakdown, recommendations)
    try:
        log_scoring_event(response, request_id=request_id, mode="text")
    except Exception:
        pass
    return response

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        # Generate MP3 in memory
        # Use 'com.au' or 'co.uk' for British accent
        tts = gTTS(text=request.text, lang='en', tld='co.uk')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Use only 1 worker to prevent OOM
    # Disable auto-reload to save resources
    # Increase timeout-keep-alive to allow model loading
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1, log_level="debug", timeout_keep_alive=120)
