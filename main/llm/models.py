"""LLM setup and agent configuration for LangChain 1.0.4."""

import logging
from langchain_openai import ChatOpenAI
from llm.prompts import SYSTEM_PROMPT, INTENT_PROMPT, QUERY_REWRITE_PROMPT
from remedies.tools import ALL_TOOLS
from config import LLM_MODEL, MAX_TOKENS

logger = logging.getLogger(__name__)


def get_llm():
    """Initialize LLM."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=MAX_TOKENS
    )


def create_remedy_agent():
    """
    Create agent for LangChain 1.0.4 using bind_tools.
    Routes queries to the appropriate remedy search tool.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    class RemedyAgent:
        def __init__(self, llm_with_tools, tools):
            self.llm = llm_with_tools
            self.tools_map = {tool.name: tool.func for tool in tools}
        
        def invoke(self, input_dict):
            """Process query and route to appropriate tool."""
            user_query = input_dict.get("input", "")
            
            # Parse input parameters
            age_months = 36
            intent = "single_remedy"
            rewritten_query = user_query
            max_results = 5
            
            for line in user_query.split("\n"):
                if "Child's age:" in line:
                    try:
                        age_months = int(line.split(":")[-1].strip().split()[0])
                    except:
                        pass
                elif "Search intent:" in line:
                    intent = line.split(":")[-1].strip().lower()
                elif "Rewritten search:" in line:
                    rewritten_query = line.split(":", 1)[-1].strip()
                elif "Max results:" in line:
                    try:
                        max_results = int(line.split(":")[-1].strip())
                    except:
                        pass
            
            # Route to appropriate tool
            try:
                if "ingredient" in intent:
                    ingredient = rewritten_query.split()[-1] if rewritten_query else "honey"
                    result = self.tools_map["find_by_ingredient"](
                        ingredient=ingredient,
                        age_months=age_months,
                        k=max_results
                    )
                elif "single" in intent:
                    result = self.tools_map["get_best_remedy"](
                        query=rewritten_query,
                        age_months=age_months
                    )
                else:
                    result = self.tools_map["list_remedy_options"](
                        query=rewritten_query,
                        age_months=age_months,
                        k=max_results
                    )
                
                return {"output": result}
            
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return {"output": {"error": f"Error during search: {str(e)}"}}
    
    return RemedyAgent(llm_with_tools, ALL_TOOLS)


def classify_intent(user_input: str, context: str) -> str:
    """Classify user intent using LLM."""
    try:
        llm = get_llm()
        prompt = INTENT_PROMPT.format(context=context, user_input=user_input)
        
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
        
        valid_intents = ["single_remedy", "multiple_remedies", "ingredient_search", "category_search"]
        if any(valid in intent for valid in valid_intents):
            return next(v for v in valid_intents if v in intent)
        
        return "single_remedy"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "single_remedy"


def rewrite_query(user_input: str, context: str) -> str:
    """Rewrite user query for better search."""
    try:
        llm = get_llm()
        prompt = QUERY_REWRITE_PROMPT.format(context=context, user_input=user_input)
        
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        return user_input