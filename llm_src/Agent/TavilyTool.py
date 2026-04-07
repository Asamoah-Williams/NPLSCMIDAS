from langchain_tavily import TavilySearch
from typing import Annotated
from langgraph.types import Command
from .State import State
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from ..utils.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()

def load_tavily_search_tool(tavily_search_max_results: int):
    """
    This function initializes a Tavily search tool, which performs searches and returns results
    based on user queries. The `max_results` parameter controls how many search results are
    retrieved for each query.

    The Tavily search tool is used to search external sources if sufficient data cannot be found using the other tools.

    Args:
        tavily_search_max_results (int): The maximum number of search results to return for each query.

    Returns:
        TavilySearchResults: A configured instance of the Tavily search tool with the specified `max_results`.
    """
    
    return TavilySearch(max_results=tavily_search_max_results)
    


@tool
def tavily_tool(state:State, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
        The Tavily search tool is used to search external sources if sufficient data cannot be found using the other tools.
    """
    user_question = state.get("user_question", None)

    content = []
    tavily = load_tavily_search_tool(tavily_search_max_results=TOOLS_CFG.tavily_search_max_results)
    tavily_results = tavily.invoke(user_question)
    results = tavily_results["results"]

    for r in results:
        content.append(r["content"])

    return Command(update={
                "results": content,
                "messages": [ToolMessage("SQL tool successful", tool_call_id=tool_call_id)]
            })