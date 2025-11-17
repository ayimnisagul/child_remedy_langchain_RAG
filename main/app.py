#streamlit app for child home remedies using LangChain RAG
import streamlit as st
import logging
import re
import json
from typing import Optional
import asyncio

from config import DEFAULT_AGE_MONTHS, CONTEXT_ROUNDS
from utils.safety import validate_input, check_moderation
from utils.formatters import render_remedy, render_remedies_list
from llm.models import create_remedy_agent, classify_intent, rewrite_query
from remedies.vectordb import load_vectordb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Child Home Remedies Assistant",
    page_icon="🧸",
    layout="wide"
)

st.title("🧸 Child Home Remedies Assistant")
st.markdown("""
**Natural, safe home remedies for common childhood ailments** – backed by research and traditional wisdom.

⚠️ These remedies are complementary, **not medical advice**.  
Always consult a pediatrician for infants under 6 months or for severe or persistent symptoms.
""")

# ========== HELPERS ==========
def extract_age_in_months(text: str) -> int:
    """Extract age in months from text."""
    text_lower = text.lower()
    
    for pattern in [r"(\d+)\s*(?:year|yr)s?\s*old", r"my\s*(\d+)\s*(?:year|yr)"]:
        if m := re.search(pattern, text_lower):
            return int(m.group(1)) * 12
    
    for pattern in [r"(\d+)\s*(?:month|mo)s?\s*old", r"(\d+)\s*(?:month|mo)s?"]:
        if m := re.search(pattern, text_lower):
            return int(m.group(1))
    
    if any(w in text_lower for w in ["infant", "baby", "newborn"]):
        return 6
    if "toddler" in text_lower:
        return 24
    
    return DEFAULT_AGE_MONTHS


def get_recent_context(n: int = CONTEXT_ROUNDS) -> str:
    """Get last n conversation rounds as string."""
    msgs = st.session_state.chat_history[-(n * 2):]
    lines = []
    
    for msg in msgs:
        if msg["role"] == "user":
            lines.append(f"Parent: {msg['text']}")
        elif msg["role"] == "assistant" and "text" in msg:
            lines.append(f"Assistant: {msg['text']}")
    
    return "\n".join(lines).strip()


# ========== SESSION STATE ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "kb_loaded" not in st.session_state:
    with st.spinner("📚 Loading remedy database..."):
        try:
            load_vectordb()
            st.session_state.kb_loaded = True
            st.success("✅ Knowledge base ready!")
        except Exception as e:
            logger.error(f"Failed to load FAISS: {e}")
            st.error(f"❌ Failed to load database: {e}")
            st.session_state.kb_loaded = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("Child's Age")
    age_mode = st.radio("Input method:", ["Extract from message", "Manual input"], index=0)
    
    age_months_manual = None
    if age_mode == "Manual input":
        age_years = st.slider("Age (years)", 0, 18, 3)
        age_months_manual = age_years * 12
        st.caption(f"= {age_months_manual} months")
    
    st.markdown("---")
    st.subheader("Search Settings")
    max_results = st.slider("Max results", 1, 10, 5)
    
    st.markdown("---")
    st.subheader("🔍 Quick Searches")
    
    if st.button("🫁 Respiratory (cough, cold)"):
        st.session_state.quick_search = "respiratory remedies for cough and cold"
    if st.button("🍽️ Digestive (stomach)"):
        st.session_state.quick_search = "digestive stomach remedies"
    if st.button("🌡️ Fever"):
        st.session_state.quick_search = "fever remedies"
    if st.button("🧴 Skin (rash, acne)"):
        st.session_state.quick_search = "skin remedies for rash"
    
    st.markdown("---")
    st.subheader("💾 Actions")
    
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("📥 Export chat"):
        chat_json = json.dumps(st.session_state.chat_history, indent=2)
        st.download_button(
            "Download JSON",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json"
        )

# ========== CHAT DISPLAY ==========
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if "remedy" in msg:
            render_remedy(msg["remedy"])
        elif "remedies" in msg:
            if msg.get("text"):
                st.markdown(msg["text"])
            render_remedies_list(msg["remedies"])
        else:
            st.markdown(msg.get("text", ""))

# ========== USER INPUT ==========
user_input = st.chat_input("Describe your child's symptom...")

# Handle quick search
if "quick_search" in st.session_state:
    user_input = st.session_state.quick_search
    del st.session_state.quick_search

if user_input:
    user_input = user_input.strip()
    
    # Validate input
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        st.warning(error_msg)
        st.stop()
    
    # Check moderation
    with st.spinner("🔍 Checking input..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            is_safe, mod_message = loop.run_until_complete(check_moderation(user_input))
            
            if not is_safe:
                st.warning(mod_message)
                st.stop()
        except Exception as e:
            logger.error(f"Moderation check error: {e}")
            # Continue on error (fail open)
    
    # Verify KB loaded
    if not st.session_state.kb_loaded:
        st.error("❌ Knowledge base not loaded. Please refresh.")
        st.stop()
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    
    # Extract or use manual age
    age_months = age_months_manual if age_mode == "Manual input" else extract_age_in_months(user_input)
    
    # Get context for LLM
    context = get_recent_context()
    
    # Classify intent and rewrite query
    with st.spinner("🤖 Understanding your question..."):
        intent = classify_intent(user_input, context)
        rewritten_query = rewrite_query(user_input, context)
        logger.info(f"Intent: {intent}, Query: {rewritten_query}")
    
    # Use agent to search remedies
    # Use agent to search remedies
    with st.spinner("🔍 Searching remedies..."):
        try:
            agent = create_remedy_agent()
            
            agent_input = f"""
            Child's age: {age_months} months
            Parent's query: {user_input}
            Rewritten search: {rewritten_query}
            Search intent: {intent}
            Max results: {max_results}
            
            Based on the intent and search query, use the appropriate tool to find remedies.
            If intent is 'single_remedy', use get_best_remedy.
            If intent is 'multiple_remedies', use list_remedy_options.
            If intent is 'ingredient_search', use find_by_ingredient.
            """
            
            agent_response = agent.invoke({
                "input": agent_input,
                "chat_history": []
            })
            
            logger.info(f"Agent response: {agent_response}")
            
            # Parse agent output and add to history
            if isinstance(agent_response, dict) and "output" in agent_response:
                output = agent_response["output"]
                
                # Handle error response
                if isinstance(output, dict) and "error" in output:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": output["error"]
                    })
                # Handle single remedy (dict with title)
                elif isinstance(output, dict) and "title" in output:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "remedy": output
                    })
                # Handle multiple remedies (list)
                elif isinstance(output, list):
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": f"Found {len(output)} remedies for '{rewritten_query}':",
                        "remedies": output
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "text": str(output)
                    })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": "I couldn't find appropriate remedies. Please try different search terms."
                })
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "text": f"An error occurred: {str(e)}. Please try again."
            })