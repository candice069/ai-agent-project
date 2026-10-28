import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY") or OPENAI_API_KEY

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")

# SerpApi Configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# TTS Configuration
TTS_SERVICE = os.getenv("TTS_SERVICE", "coqui")  # coqui, google, azure, elevenlabs
TTS_API_KEY = os.getenv("TTS_API_KEY")
COQUI_MODEL = os.getenv("COQUI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
COQUI_SPEAKER_WAV = os.getenv("COQUI_SPEAKER_WAV")
COQUI_LANGUAGE = os.getenv("COQUI_LANGUAGE", "en")

# STT Configuration
STT_SERVICE = os.getenv("STT_SERVICE", "whisper")  # whisper, google, azure
STT_API_KEY = os.getenv("STT_API_KEY")

# RAG Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 3072))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 5))
RAG_CANDIDATE_LIMIT = int(os.getenv("RAG_CANDIDATE_LIMIT", 20))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", 0.45))
RAG_DENSE_WEIGHT = float(os.getenv("RAG_DENSE_WEIGHT", 0.65))

# Memory Configuration
MEMORY_RECENT_MESSAGES = int(os.getenv("MEMORY_RECENT_MESSAGES", 10))
MEMORY_SUMMARY_TRIGGER = int(os.getenv("MEMORY_SUMMARY_TRIGGER", 16))
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", 60 * 60 * 24 * 30))

# Realtime Configuration
WEBSOCKET_SYNTHESIZE_AUDIO = os.getenv("WEBSOCKET_SYNTHESIZE_AUDIO", "false").lower() == "true"
