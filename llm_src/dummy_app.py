# from src.chatbot import ChatBot
from chatbot import respond

# c = ChatBot()
user_id = "u_200"
session_id = "s_2025_09_15_01"
# thread_id = f"{user_id}:{session_id}"

# res = c.respond(message="what is mean squared error", user_id=user_id, session_id=session_id)
res = respond(message="what is the latest news in finance", user_id=user_id, session_id=session_id)
print(res)