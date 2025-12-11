"""Configuration management for API keys and model settings."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")

# Model Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")  # Default to gemini-pro

# RAG Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Debate Configuration
MAX_DEBATE_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "2"))

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
RAW_PDFS_DIR = os.path.join(DATA_DIR, "raw_pdfs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

