from .State import State

def reset_state(state: State) -> State:
    """
    Resets the state variables to their initial values while preserving the messages.

    Args:
        state(State): Current state of the graph (does not need to be passed in manually)
    Return: 
        state(State): A dictionary with reset state variables and preserved messages
    """
    return {
        "user_question": "",
        "sql_query": "",
        "results": [],
        "sql_results": [],
        "visualization": "",
        "answer": "",
        "visualization_reason": "",
        "formatted_data_for_visualization": {}, 
        "error": "",
        "answer_done": False,
        "messages": state.get("messages", [])
    }
    

def capture_node(state: State):
    """
    Executes the primary language model with tools bound and returns the generated message.

    Args:
        state(State): Current state of the graph (does not need to be passed in manually)

    Return:
        reset(dict): A dictionary with state variables as the key

    """
    clean_state = reset_state(state)

    if state.get("messages"):
        latest_message = state["messages"][-1]
        if hasattr(latest_message, 'content'):
            clean_state["user_question"] = latest_message.content
        else:
            clean_state["user_question"] = latest_message.get("content", "")

    return clean_state