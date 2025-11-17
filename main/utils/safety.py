"""Input validation and moderation using OpenAI Moderation API."""

import logging
from typing import Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI()


def validate_input(user_input: str) -> Tuple[bool, str]:
    """
    Validate user input for length and content safety.
    
    Returns:
        (is_valid: bool, message: str)
    """
    from config import MIN_INPUT_LENGTH, MAX_INPUT_LENGTH
    
    user_input = user_input.strip()
    
    # Length validation
    if len(user_input) < MIN_INPUT_LENGTH:
        return False, f"⚠️ Input too short. Please describe the symptom in more detail (min {MIN_INPUT_LENGTH} characters)."
    
    if len(user_input) > MAX_INPUT_LENGTH:
        return False, f"⚠️ Input too long. Please use under {MAX_INPUT_LENGTH} characters."
    
    return True, ""


async def check_moderation(user_input: str) -> Tuple[bool, str]:
    """
    Check input against OpenAI Moderation API for harmful content.
    
    Returns:
        (is_safe: bool, message: str)
    """
    try:
        response = client.moderations.create(input=user_input)
        
        if response.results[0].flagged:
            flagged_categories = [
                cat for cat, flagged in response.results[0].category_scores.items()
                if getattr(response.results[0], cat)
            ]
            logger.warning(f"Moderation flagged input: {flagged_categories}")
            return False, "⚠️ Your input contains inappropriate content. Please rephrase your question."
        
        return True, ""
    except Exception as e:
        logger.error(f"Moderation check failed: {e}")
        # Fail open - allow if moderation service fails
        return True, ""