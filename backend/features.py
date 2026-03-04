import librosa
import numpy as np
import torch
import os

_ssl_model_cache = {}


def _get_ssl_model(model_name: str = "facebook/wav2vec2-base"):
    """Lazy-load and cache the SSL encoder + processor so it's loaded once per process."""
    if model_name in _ssl_model_cache:
        return _ssl_model_cache[model_name]
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _ssl_model_cache[model_name] = (processor, model)
        return processor, model
    except Exception as e:
        import traceback
        print(f"SSL model load failed ({model_name}): {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


def extract_ssl_features(
    y: np.ndarray,
    sr: int = 16000,
    model_name: str = "facebook/wav2vec2-base",
    max_duration: float = 60.0,
):
    """
    Extract self-supervised speech representations from a waveform.

    Returns:
        ssl_tensor: (1, Time', hidden_dim) — e.g. (1, T, 768) for wav2vec2-base
        Returns None on failure.
    """
    try:
        max_samples = int(sr * max_duration)
        if len(y) > max_samples:
            y = y[:max_samples]

        processor, model = _get_ssl_model(model_name)
        if processor is None or model is None:
            return None

        inputs = processor(
            y, sampling_rate=sr, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.last_hidden_state  # (1, Time', 768)
    except Exception as e:
        import traceback
        print(f"SSL feature extraction failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def extract_ssl_features_from_path(
    audio_path: str,
    sr: int = 16000,
    model_name: str = "facebook/wav2vec2-base",
    max_duration: float = 60.0,
):
    """Convenience wrapper that loads audio from a file path."""
    try:
        if not os.path.exists(audio_path):
            print(f"SSL extraction skipped — file not found: {audio_path}")
            return None
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration)
        if y is None or len(y) == 0:
            print(f"SSL extraction skipped — empty waveform: {audio_path}")
            return None
        return extract_ssl_features(y, sr=sr, model_name=model_name, max_duration=max_duration)
    except Exception as e:
        import traceback
        print(f"SSL feature extraction from path failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

def _extract_features_core(y: np.ndarray, sr: int = 16000, n_mfcc: int = 40, n_fft: int = 2048, hop_length: int = 512):
    """
    Core feature extraction from an in-memory waveform.
    """
    if y is None or len(y) == 0:
        return None, None, None

    # --- 1. Spectral Features (MFCC) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T  # (Time, MFCC)

    # Standardize MFCC (Zero Mean, Unit Variance) per utterance
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-6)

    # --- 2. Prosodic Features (F0, Energy) ---
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr, frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0)  # Replace NaNs with 0 (unvoiced)

    # Energy (RMSE)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # Combine Prosody (Time, 2)
    min_len = min(len(mfcc), len(f0), len(rms))
    prosody = np.stack([f0[:min_len], rms[:min_len]], axis=1)
    mfcc = mfcc[:min_len]

    # Normalize Prosody
    prosody = (prosody - np.mean(prosody, axis=0)) / (np.std(prosody, axis=0) + 1e-6)

    # --- 3. Static Features (Heuristic) ---
    duration = len(y) / sr
    speech_energy_threshold = np.mean(rms) * 0.5
    speech_frames = rms > speech_energy_threshold
    speech_duration = np.sum(speech_frames) * hop_length / sr

    static_feats = np.array([
        duration,
        speech_duration / (duration + 1e-6),
        np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0,
        np.std(f0[f0 > 0]) if np.any(f0 > 0) else 0,
        np.max(f0),
        np.mean(rms),
        np.std(rms),
        np.max(rms),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    ])

    static_feats[0] /= 60.0  # Duration in minutes
    static_feats[2] /= 200.0  # F0 ~ 200Hz

    return (
        torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0),  # (1, Time, 40)
        torch.tensor(prosody, dtype=torch.float32).unsqueeze(0),  # (1, Time, 2)
        torch.tensor(static_feats, dtype=torch.float32).unsqueeze(0)  # (1, 10)
    )


def extract_features_from_waveform(y: np.ndarray, sr: int = 16000, n_mfcc: int = 40, n_fft: int = 2048, hop_length: int = 512):
    """
    Public helper for training-time augmentation flows that already have waveform data.
    """
    try:
        return _extract_features_core(y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    except Exception as e:
        print(f"Feature extraction from waveform failed: {e}")
        return None, None, None


def extract_features(audio_path: str, sr: int = 16000, n_mfcc: int = 40, n_fft: int = 2048, hop_length: int = 512):
    """
    Extracts multi-modal features for the Deep Learning Scoring Model.
    
    Returns:
        mfcc_tensor: (Time, n_mfcc)
        prosody_tensor: (Time, 2) [F0, Energy]
        static_features: (10,) [WPM, Pause Rate, etc.]
    """
    try:
        # Optimization: Limit to first 60 seconds for training efficiency
        # IELTS videos can be long, but 60s is enough to capture pronunciation/fluency stats
        y, _ = librosa.load(audio_path, sr=sr, duration=60)
        return _extract_features_core(y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None, None, None
