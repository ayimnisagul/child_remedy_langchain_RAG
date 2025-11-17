"""LangChain tool definitions for remedy search."""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from remedies.search import search_remedies
import logging

logger = logging.getLogger(__name__)


class RemedyQuery(BaseModel):
    query: str = Field(..., description="Symptom or condition description")
    age_months: Optional[int] = Field(36, description="Child's age in months")


class RemediesQuery(BaseModel):
    query: str = Field(..., description="Symptom or condition")
    age_months: Optional[int] = Field(36, description="Child's age in months")
    k: Optional[int] = Field(5, description="Number of results to return")


class IngredientQuery(BaseModel):
    ingredient: str = Field(..., description="Ingredient name (e.g., 'honey', 'ginger')")
    age_months: Optional[int] = Field(36, description="Child's age in months")
    k: Optional[int] = Field(5, description="Number of results")


def get_single_remedy(query: str, age_months: Optional[int] = 36) -> Dict:
    """Get the single best remedy for a symptom."""
    if age_months is None:
        age_months = 36
    
    try:
        remedies = search_remedies(query, age_months, k=1)
    except Exception as e:
        logger.error(f"Error searching remedies: {e}")
        return {"error": "An error occurred while searching. Please try again."}
    
    if not remedies:
        if age_months < 12:
            return {
                "error": f"⚠️ No safe remedies for {age_months}-month-old. Many contain honey or other unsafe ingredients for infants under 12 months. Consult a pediatrician."
            }
        return {"error": f"No remedies found for '{query}'. Try different search terms."}
    
    remedy = remedies[0]
    
    if age_months < remedy["age_min_months"] + 6:
        remedy["safety_note"] = f"⚠️ This is newly appropriate for {age_months} months. Start with small amounts and monitor."
    
    return remedy


def list_multiple_remedies(query: str, age_months: Optional[int] = 36, k: int = 5) -> List[Dict]:
    """List multiple remedies for a condition."""
    if age_months is None:
        age_months = 36
    
    remedies = search_remedies(query, age_months, k=k)
    
    if not remedies and age_months < 12:
        return [{
            "error": f"⚠️ Limited safe remedies for infants under 12 months. Consult pediatrician."
        }]
    
    return remedies


def find_remedies_by_ingredient(ingredient: str, age_months: Optional[int] = 36, k: int = 5) -> List[Dict]:
    """Find remedies using a specific ingredient."""
    if age_months is None:
        age_months = 36
    
    query = f"{ingredient} remedy ingredient"
    remedies = search_remedies(query, age_months, k=k * 2)
    
    filtered = []
    ingredient_lower = ingredient.lower()
    
    for remedy in remedies:
        ingredients_text = " ".join(remedy.get("ingredients", [])).lower()
        if ingredient_lower in ingredients_text:
            filtered.append(remedy)
    
    return filtered[:k]


# Create LangChain tools
GET_REMEDY_TOOL = StructuredTool.from_function(
    func=get_single_remedy,
    name="get_best_remedy",
    description="Get the single best home remedy for a child's symptom. Use when parent asks for one specific remedy.",
    args_schema=RemedyQuery,
)

LIST_REMEDIES_TOOL = StructuredTool.from_function(
    func=list_multiple_remedies,
    name="list_remedy_options",
    description="List multiple home remedies for a condition. Use when parent asks for options or alternatives.",
    args_schema=RemediesQuery,
)

FIND_INGREDIENT_TOOL = StructuredTool.from_function(
    func=find_remedies_by_ingredient,
    name="find_by_ingredient",
    description="Find remedies using a specific ingredient. Use when parent has an ingredient and wants to know what to make with it.",
    args_schema=IngredientQuery,
)

ALL_TOOLS = [GET_REMEDY_TOOL, LIST_REMEDIES_TOOL, FIND_INGREDIENT_TOOL]

