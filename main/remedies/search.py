"""Remedy search logic and ranking."""

import logging
import re
from typing import List, Dict
from remedies.vectordb import get_vectordb
from config import DEFAULT_AGE_MONTHS

logger = logging.getLogger(__name__)


def search_remedies(query: str, age_months: int = DEFAULT_AGE_MONTHS, k: int = 10) -> List[Dict]:
    """
    Search and return age-appropriate remedies with ranking.
    
    Args:
        query: Symptom/condition description
        age_months: Child's age in months
        k: Number of results to return
    
    Returns:
        List of ranked remedies
    """
    vectordb = get_vectordb()
    
    try:
        docs = vectordb.similarity_search(query, k=k * 2)
        logger.info(f"Found {len(docs)} initial documents for query '{query}'")
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []
    
    if not docs:
        logger.warning(f"No results for '{query}', trying broader query...")
        try:
            docs = vectordb.similarity_search("common home remedies for child", k=k * 2)
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    remedies = []
    for doc in docs:
        md = doc.metadata or {}
        
        # Only food remedies
        if md.get("type") != "food_remedy":
            continue
        
        # Age safety check
        age_min = md.get("age_min_months", 0)
        age_max = md.get("age_max_months")
        
        if age_months < age_min or (age_max and age_months > age_max):
            continue
        
        remedy = {
            "title": md.get("title", "Untitled"),
            "category": md.get("illness_category", "other"),
            "symptoms": md.get("specific_symptoms", []),
            "ingredients": md.get("ingredients", []),
            "steps": md.get("steps", []),
            "dosage": md.get("dosage"),
            "description": md.get("body", ""),
            "why_it_works": md.get("why_it_works", ""),
            "duration": md.get("duration"),
            "age_min_months": age_min,
            "age_max_months": age_max,
            "contains_honey": md.get("contains_honey", False),
            "contraindications": md.get("contraindications", []),
            "warnings": md.get("warnings", []),
            "source_url": md.get("source_url"),
            "trust_score": md.get("trust_score", "medium"),
            "evidence_level": md.get("evidence_level", "traditional")
        }
        
        score = _calculate_relevance(remedy, query, age_months)
        remedy["_score"] = score
        remedies.append(remedy)
    
    remedies.sort(key=lambda x: x["_score"], reverse=True)
    
    for r in remedies:
        r.pop("_score", None)
    
    return remedies[:k]


def _calculate_relevance(remedy: Dict, query: str, age_months: int) -> float:
    """Calculate relevance score for ranking."""
    score = 1.0
    
    trust_weights = {"high": 1.5, "medium": 1.0, "low": 0.7}
    score *= trust_weights.get(remedy["trust_score"], 1.0)
    
    evidence_weights = {"research-backed": 1.3, "anecdotal": 1.0, "traditional": 1.1}
    score *= evidence_weights.get(remedy["evidence_level"], 1.0)
    
    if age_months >= remedy["age_min_months"] * 1.5:
        score *= 1.2
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))
    symptom_words = set()
    
    for symptom in remedy.get("symptoms", []):
        symptom_words.update(re.findall(r'\w+', symptom.lower()))
    
    overlap = len(query_words & symptom_words)
    if overlap > 0:
        score *= (1 + overlap * 0.3)
    
    return score
