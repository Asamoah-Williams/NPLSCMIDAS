from typing import List, Any, Annotated, Dict, NotRequired
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages


def add_results(existing: List[Any], new: List[Any]) -> List[Any]:
    if existing is None:
        existing = []
    if new is None:
        new = []
    return existing + new


def add_to_dict(existing: Dict[str, Any], new: Dict[str, any]):
    if existing is None:
        existing = {}
    if new is None:
        new = {}
    return {**existing, **new}


# State class to hold the agent's state
class State(AgentState):
    user_question: NotRequired[str]  # user's question
    sql_query: NotRequired[str]      # sql query from sql tool
    results: NotRequired[Annotated[List[Any], add_results]]  # results from tools
    sql_results: NotRequired[Annotated[List[Any], add_results]]  # results from sql tool
    visualization: NotRequired[str]
    answer: NotRequired[str]
    visualization_reason: NotRequired[str]  # reason for a certain visualization type
    formatted_data_for_visualization: NotRequired[Dict[str, Any]]  # data for visualization
    error: NotRequired[str]  # any error that comes up
    answer_done: NotRequired[bool]
    messages: Annotated[list, add_messages]
