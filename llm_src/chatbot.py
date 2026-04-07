# from typing import List, Tuple
# from .utils.load_project_configs import LoadProjectConfigs
# from .utils.load_tools_config import LoadToolsConfig
# from .Agent.graph import build_graph

# from typing import List, Tuple
# from utils.load_project_configs import LoadProjectConfigs
from .utils.load_project_configs import LoadProjectConfigs
# from utils.load_tools_config import LoadToolsConfig
from .utils.load_tools_config import LoadToolsConfig
from .Agent.graph import build_graph

PROJECT_CFG = LoadProjectConfigs()
TOOLS_CFG = LoadToolsConfig()

graph = build_graph()


# class ChatBot:
#     """
#     A class to handle chatbot interactions by utilizing a pre-defined agent graph. The chatbot processes
#     user messages, generates appropriate responses, and saves the chat history to a specified memory directory.

#     Attributes:
#         config (dict): A configuration dictionary that stores specific settings such as the `thread_id`.

#     Methods:
#         respond(chatbot: List, message: str) -> Tuple:
#             Processes the user message through the agent graph, generates a response, appends it to the chat history,
#             and writes the chat history to a file.
#     """
    # @staticmethod
def respond(message: str, user_id: str, session_id: str) -> dict:
    """
    Processes a user message using the agent graph, generates a response, and appends it to the chat history.
    The chat history is also saved to a memory file for future reference.

    Args:
        chatbot (List): A list representing the chatbot conversation history. Each entry is a tuple of the user message and the bot response.
        message (str): The user message to process.

    Returns:
        Tuple: Returns an empty string (representing the new user input placeholder) and the updated conversation history.
    """

    input_msg = {"role": "user", "content": message}
    
    final_state = graph.invoke(
        {"messages": [input_msg]},
        config={"configurable": {
            "thread_id": f"{user_id}:{session_id}",
            "checkpoint_ns": f"myapp:{user_id}"   # all sessions for this user live here
        }}
    )

    payload = {
    "visualization": final_state.get("visualization"),
    "formatted_data_for_visualization": final_state.get("formatted_data_for_visualization"),
    "answer": final_state.get("answer"),
    "answer_done": final_state.get("answer_done"),
    }

    return payload
