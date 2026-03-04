# IELTS Speaking Coach

An AI-powered IELTS Speaking practice tool that records your response, transcribes it, scores it across all four IELTS criteria, and provides a model answer, vocabulary upgrades, and grammar coaching -- all in real time.

## Features

- **Audio recording and transcription** via Whisper ASR
- **IELTS band scoring** with breakdown across Fluency & Coherence, Lexical Resource, Grammar, and Pronunciation
- **Model answer generation** at your level + 1 band (i+1 principle) using spoken English register
- **Context-aware vocabulary recommendations** that match the topic you're discussing
- **Grammar coaching** with corrections and enhancement suggestions
- **Ontology-driven learning trajectory** that tracks your progress and personalizes recommendations over time
- **Text-to-speech** for model answer playback
- **Vue 3 web frontend** -- works on any browser, mobile or desktop

## Architecture

```
Browser (Vue 3 SPA)
    |
    |  /api/* requests
    v
  Nginx (SSL + static files + reverse proxy)
    |
    v
  FastAPI backend (:8080)
    |
    +-- Whisper ASR (transcription)
    +-- spaCy NLP (text analysis)
    +-- Rule-based scoring engine
    +-- LLM evaluator (local Qwen3 or DeepSeek API)
    +-- Ontology graph (vocabulary relationships)
    +-- SQLite persistence (learner state)
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- ffmpeg (for audio processing)
- ~8 GB RAM (for Whisper + spaCy models)

## Quick Start

### Backend

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ielts-speaking-coach.git
cd ielts-speaking-coach

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md

# Configure environment
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY (optional -- needed only if no local model)

# Start the backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

For production build:

```bash
npm run build
# Deploy dist/ folder to your web server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | (empty) | DeepSeek API key for LLM fallback |
| `LOCAL_MODEL_PATH` | (empty) | Path to fine-tuned PEFT adapter |
| `LOCAL_ENABLE_REASONING` | `0` | Enable thinking mode for local model |
| `ONTOLOGY_TRAJECTORY_ENABLED` | `0` | Enable ontology-driven learning trajectory |
| `TRAJECTORY_DB_PATH` | `backend/data/learner_trajectory.db` | SQLite path for learner state persistence |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze_audio` | Upload audio for async analysis |
| `GET` | `/task_status/{task_id}` | Poll analysis status |
| `POST` | `/analyze` | Analyze text directly (no audio) |
| `POST` | `/analyze_audio_sync` | Synchronous audio analysis |
| `POST` | `/feedback_recommendation` | Accept/ignore vocabulary suggestion |
| `POST` | `/user/word_used` | Log successful word usage |
| `GET` | `/user/profile/{user_id}` | Get user profile and trajectory |
| `GET` | `/user/progress/{user_id}` | Get learning progress |
| `POST` | `/tts` | Text-to-speech for model answers |

## Deployment

### Nginx + SSL

1. Build the frontend: `cd frontend && npm run build`
2. Copy `dist/` contents to `/var/www/speakingcoach/`
3. Copy `frontend/nginx-speakingcoach.conf` to `/etc/nginx/sites-available/speaking-coach`
4. Edit the config: replace `your-domain.example.com` and SSL cert paths
5. Enable: `ln -s /etc/nginx/sites-available/speaking-coach /etc/nginx/sites-enabled/`
6. Test and reload: `nginx -t && nginx -s reload`

### Backend as a Service

Use supervisord or systemd to keep the backend running:

```ini
# /etc/supervisor/conf.d/speakingcoach.conf
[program:speakingcoach]
directory=/path/to/ielts-speaking-coach
command=/path/to/venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/speakingcoach/app.log
environment=PYTHONUNBUFFERED="1"
```

## Project Structure

```
ielts-speaking-coach/
  backend/
    main.py                   # FastAPI app, scoring, recommendations
    llm_evaluator.py          # LLM-based model answers and vocabulary
    vocabulary_ontology.py    # Ontology graph for concept relationships
    persistence.py            # SQLite learner state persistence
    trajectory_planner.py     # Learning trajectory planning
    transcribe.py             # Whisper ASR subprocess
    features.py               # Audio feature extraction
    models/
      scoring_model.py        # Deep learning scoring model
      unisep_scorer.py        # Pronunciation scoring
    data/
      ontology_seed.json      # Curated prerequisite relationships
    tests/
      test_ontology.py
      test_persistence.py
      test_trajectory_planner.py
      test_api_contracts.py
  frontend/
    src/
      App.vue                 # Main application layout
      api.js                  # API client
      composables/
        useRecorder.js        # Browser audio recording
      components/
        PartSelector.vue
        RecordingControls.vue
        ResultCard.vue
        ScoreBar.vue
        VocabCard.vue
        GrammarCard.vue
        ModelAnswer.vue
    nginx-speakingcoach.conf  # Production nginx config template
  requirements.txt
  .env.example
  LICENSE
```

## License

MIT
