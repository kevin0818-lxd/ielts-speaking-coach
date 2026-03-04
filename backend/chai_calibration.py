import numpy as np

class ChaiCalibrator:
    """
    Implements the CHAI hybrid calibration framework (Polasa et al., 2025).
    Calibrates model scores using a human prior and monotone regression.
    
    Formula: y_hat = clip(w0 + w1 * y_m + w2 * h_prior, 0, 9)
    """
    
    def __init__(self):
        # Human Priors (h_bar) - Estimated anchors from "average" learner performance
        # In a real deployment, these are calculated from a dev set of human-rated transcripts.
        # Here we use conservative estimates based on typical EFL learner profiles (Band 5.5-6.0 avg).
        self.global_prior = 6.0
        
        # Scenario-specific priors (b_scene)
        # Part 1 is usually easier (higher prior), Part 3 is harder (lower prior).
        self.scene_priors = {
            1: 0.2,  # Part 1: ~6.2
            2: 0.0,  # Part 2: ~6.0
            3: -0.3  # Part 3: ~5.7
        }
        
        # Criterion-specific offsets (mu_c)
        # Pronunciation often scores lower for EFL learners than Grammar/Fluency.
        self.criterion_priors = {
            "fluency": 0.0,
            "lexical": -0.1,
            "grammar": 0.1,
            "pronunciation": -0.2
        }
        
        # Calibration Weights (w0, w1, w2)
        # w1 (Model Weight): High trust in model (0.8 - 0.9)
        # w2 (Prior Weight): Low but non-zero to stabilize outliers (0.1 - 0.2)
        # w0 (Bias): Small adjustment
        self.weights = {
            "fluency": {"w0": 0.1, "w1": 0.85, "w2": 0.15},
            "lexical": {"w0": 0.0, "w1": 0.90, "w2": 0.10},
            "grammar": {"w0": 0.0, "w1": 0.85, "w2": 0.15},
            "pronunciation": {"w0": 0.2, "w1": 0.80, "w2": 0.20}
        }

    def get_prior(self, part, criterion):
        """
        Calculates the Human Prior (h_bar) for a given scenario and criterion.
        h_bar = global_prior + scene_prior + criterion_prior
        """
        h = self.global_prior + self.scene_priors.get(part, 0.0) + self.criterion_priors.get(criterion, 0.0)
        return max(1.0, min(9.0, h))

    def calibrate(self, model_score, part, criterion):
        """
        Applies the monotone calibration formula.
        """
        if criterion not in self.weights:
            return model_score
            
        w = self.weights[criterion]
        h_prior = self.get_prior(part, criterion)
        
        # y_hat = w0 + w1*y_m + w2*h_prior
        calibrated_score = w["w0"] + (w["w1"] * model_score) + (w["w2"] * h_prior)
        
        # Clip to valid band range
        return max(1.0, min(9.0, calibrated_score))

# Singleton instance
_calibrator = ChaiCalibrator()

def calibrate_score(score, part, criterion):
    return _calibrator.calibrate(score, part, criterion)
