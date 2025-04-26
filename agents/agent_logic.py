from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Summarizer using context
def summarize(content: str) -> str:
    prompt = ChatPromptTemplate([
        ("system", "You are a professional summarizer. Provide a clear, concise summary."),
        ("human", f"Summarize the following content:\n\n{content}")
    ])
    print("Prompt for summarization:", prompt)
    return prompt
    """chain = prompt | llm
    return chain.invoke({"content": content})"""

# KPI Extractor
def extract_kpis(content: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst. Extract and list key KPIs and numeric metrics."),
        ("human", f"Extract KPIs and important numbers from the following content:\n\n{content}")
    ])
    return prompt
    """chain = prompt | llm
    return chain.invoke({"content": content})"""

# Report Generation based on context and topic
def generate_report(context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business report writer. Write a structured, formal report."),
        ("human", f"Based on the following context, generate a brief report:\n\n{context}")
    ])
    return prompt
    """chain = prompt | llm
    return chain.invoke({"topic": topic, "context": context})"""

# Simulated Web Search (for web-related queries)
def search_web(query: str) -> str:
    fake_web_results = {
        "ESG risks 2024": "Recent ESG risks include regulatory changes in Europe, increased carbon reporting requirements, and focus on biodiversity.",
        "AI market trends 2025": "The AI market is expected to grow by 35% CAGR, with dominance in healthcare, finance, and education sectors.",
    }
    return fake_web_results.get(query, f"Simulated web search results for '{query}' not found.")

# Register all the tools for use
from agents.tool_agent import register_tools
register_tools({
    "summarize": summarize,
    "extract_kpis": extract_kpis,
    "generate_report": generate_report,
    "search_web": search_web,
})
