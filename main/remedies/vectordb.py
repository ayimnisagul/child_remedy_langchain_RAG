"""FAISS vectordb operations."""

import logging
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from config import FAISS_PATH, EMBED_MODEL

logger = logging.getLogger(__name__)

_vectordb: Optional[FAISS] = None


def load_vectordb() -> FAISS:
    """Load and cache FAISS vectordb."""
    global _vectordb
    
    if _vectordb is None:
        try:
            embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
            _vectordb = FAISS.load_local(
                str(FAISS_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ FAISS vectordb loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FAISS database: {e}")
            raise
    
    return _vectordb


def get_vectordb() -> FAISS:
    """Get cached vectordb instance."""
    return load_vectordb()