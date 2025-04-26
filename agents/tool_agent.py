# agents/tool_agent.py

from typing import Callable, Dict

# Placeholder for actual tools
TOOLS: Dict[str, Callable] = {}

def register_tools(tool_dict: Dict[str, Callable]):
    """Register external tool functions."""
    global TOOLS
    TOOLS = tool_dict


def decide_and_act(user_input: str, rag_chain) -> str:
    """
    Parses user input, decides which tools to use, and invokes them with context.
    """
    actions = []

    # --- Detect intents ---
    if "summarize" in user_input.lower():
        actions.append("summarize")
    if "kpi" in user_input.lower() or "compare" in user_input.lower():
        actions.append("extract_kpis")
    if "report" in user_input.lower():
        actions.append("generate_report")
    if "search" in user_input.lower() or "latest" in user_input.lower():
        actions.append("search_web")

    # --- Default: fallback to RAG if no tool matches ---
    if not actions:
        return rag_chain.invoke(user_input)

    # --- Execute each tool and collect responses ---
    combined_response = ""
    for action in actions:
        if action == "search_web":
            result = TOOLS[action](user_input)
        else:
            context = rag_chain.invoke(user_input)
            if action == "generate_report":
                result = TOOLS[action](user_input, context)
            else:
                result = TOOLS[action](context)
        combined_response += f"### {action.replace('_', ' ').title()} Result:\n{result}\n\n"
    print(f"Combined response from tools: {combined_response}")
    return combined_response
