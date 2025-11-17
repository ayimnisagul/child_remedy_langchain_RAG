"""Configuration and constants for the remedy app."""

from pathlib import Path
from typing import Final

# Paths
FAISS_PATH: Final[Path] = Path("faiss_indices/food_remedies")

# LLM Settings
LLM_MODEL: Final[str] = "gpt-4o-mini"
EMBED_MODEL: Final[str] = "text-embedding-3-small"
MAX_TOKENS: Final[int] = 300


# Search Settings
DEFAULT_RESULTS: Final[int] = 5
MAX_RESULTS: Final[int] = 10
CONTEXT_ROUNDS: Final[int] = 3

# Safety
MIN_INPUT_LENGTH: Final[int] = 3
MAX_INPUT_LENGTH: Final[int] = 500

# Remedy defaults
DEFAULT_AGE_MONTHS: Final[int] = 36

