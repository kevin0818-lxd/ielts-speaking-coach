import sys
import json
import warnings
import numpy as np
import os
import time
import uuid
import urllib.request

# Suppress warnings
warnings.filterwarnings("ignore")

def _debug_log(*args, **kwargs):
    pass


def transcribe(audio_path, model_name="base"): # Upgraded default to base
    try:
        # #region agent log
        _debug_log(
            "H0",
            "backend/transcribe.py:transcribe:entry",
            "Transcribe entry",
            {
                "audio_path": audio_path,
                "model_name": model_name,
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "audio_exists": os.path.exists(audio_path),
                "audio_size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else None,
            },
        )
        # #endregion
        import whisper
        # #region agent log
        _debug_log(
            "H1",
            "backend/transcribe.py:transcribe:post_import_whisper",
            "Whisper import metadata",
            {
                "whisper_file": getattr(whisper, "__file__", None),
                "whisper_version": getattr(whisper, "__version__", None),
                "has_load_model": hasattr(whisper, "load_model"),
            },
        )
        # #endregion
        # Load model
        # User requested enhanced accuracy, so we prioritize accuracy over speed.
        # "small" is significantly better than "base" with acceptable CPU latency.

        # #region agent log
        _debug_log(
            "H2",
            "backend/transcribe.py:transcribe:before_load_model",
            "Loading Whisper model",
            {"model_name": model_name, "device": "cpu"},
        )
        # #endregion
        model = whisper.load_model(model_name, device="cpu")

        # #region agent log
        _debug_log(
            "H2",
            "backend/transcribe.py:transcribe:after_load_model",
            "Whisper model loaded",
            {"n_mels": getattr(getattr(model, "dims", None), "n_mels", None)},
        )
        # #endregion

        # Context prompt to guide the model towards IELTS Speaking content
        # This helps with specific vocabulary and reducing hallucinations
        initial_prompt = "The following is a candidate's response in an IELTS Speaking test. The speech may contain hesitations, self-corrections, and varied accents."

        # Probe audio decode and mel creation directly to isolate failures near log_mel_spectrogram
        try:
            from whisper.audio import load_audio, log_mel_spectrogram
            audio_probe = load_audio(audio_path)
            # #region agent log
            _debug_log(
                "H3",
                "backend/transcribe.py:transcribe:audio_probe",
                "Audio probe loaded",
                {"audio_len": int(len(audio_probe)), "audio_dtype": str(getattr(audio_probe, "dtype", None))},
            )
            # #endregion
            mel_probe = log_mel_spectrogram(audio_probe, model.dims.n_mels, padding=480000)
            # #region agent log
            _debug_log(
                "H4",
                "backend/transcribe.py:transcribe:mel_probe",
                "Mel probe computed",
                {"mel_shape": list(mel_probe.shape) if hasattr(mel_probe, "shape") else None},
            )
            # #endregion
        except Exception as probe_e:
            # #region agent log
            _debug_log(
                "H3_H4",
                "backend/transcribe.py:transcribe:probe_exception",
                "Audio/mel probe failed",
                {"error": str(probe_e), "error_type": type(probe_e).__name__},
            )
            # #endregion

        # Transcribe with optimized parameters for accuracy
        result = model.transcribe(
            audio_path,
            beam_size=5, # Higher beam size for better accuracy
            best_of=5,   # Candidates to sample from
            temperature=0.0, # Deterministic (greedy) sampling
            condition_on_previous_text=False, # Independent segments
            initial_prompt=initial_prompt, # Context injection
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            fp16=False # Force FP32 for CPU stability
        )
        
        # Calculate pronunciation score
        segments = result.get("segments", [])
        pron_score = 0.5
        avg_logprob = None
        if segments:
            # Geometric mean of probabilities is a better confidence estimator
            logprobs = [s.get("avg_logprob", -1.0) for s in segments]
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                pron_score = np.exp(avg_logprob)
            
        output = {
            "text": result["text"].strip(),
            "pron_score": float(pron_score),
            "avg_logprob": float(avg_logprob) if avg_logprob is not None else None,
            "asr_model": model_name
        }
        # #region agent log
        _debug_log(
            "H5",
            "backend/transcribe.py:transcribe:success",
            "Transcribe success",
            {
                "text_len": len(output.get("text", "")),
                "pron_score": output.get("pron_score"),
                "avg_logprob": output.get("avg_logprob"),
            },
        )
        # #endregion
        print(json.dumps(output))
        
    except Exception as e:
        import traceback
        # #region agent log
        _debug_log(
            "H_FATAL",
            "backend/transcribe.py:transcribe:exception",
            "Transcribe exception",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        # #endregion
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio file provided"}))
        sys.exit(1)
        
    audio_file = sys.argv[1]
    # Default to 'small' if not specified, for better accuracy
    model = sys.argv[2] if len(sys.argv) > 2 else "small" 
    transcribe(audio_file, model)
