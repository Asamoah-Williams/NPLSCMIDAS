# llm_src/utils/bot.py
from ..chatbot import respond
from datetime import datetime
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "configs"
CFG = yaml.safe_load((ROOT / "project_config.yml").read_text())

BOT_USERNAME = CFG["chat"]["bot_username"]
BOT_DELAY_SECONDS = float(CFG["chat"]["bot_delay_seconds"])

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

def generate_bot_reply(user_text: str, user_id="u_default", session_id="s_default"):
    try:
        res = respond(user_text, user_id=user_id, session_id=session_id)
        return res.get("answer", "Sorry, I couldn’t generate an answer.")
    except Exception as e:
        return f"Error from LLM: {e}"


def bot_reply_async(socketio, room, user_text, user_id, session_id, save_message_fn=None):
    """
    Asynchronous function that generates a reply from the bot and emits + saves it.
    The save_message_fn callback (if provided) should handle persisting messages.
    """
    socketio.sleep(BOT_DELAY_SECONDS)
    reply_data = None

    try:
        reply_data = respond(user_text, user_id=user_id, session_id=session_id)
        reply_text = reply_data.get("answer", "Sorry, I couldn’t generate an answer.")
    except Exception as e:
        reply_text = f"Error from LLM: {e}"
        reply_data = {}

    # --- Persist + emit bot message ---
    if save_message_fn:
        # use the callback from chatSocket
        save_message_fn(socketio, room, BOT_USERNAME, reply_text)
    else:
        # fallback: emit only (no persistence)
        msg = {
            "id": f"{BOT_USERNAME}-{datetime.utcnow().timestamp()}",
            "username": BOT_USERNAME,
            "room": room,
            "text": reply_text,
            "meta": {},
            "ts": _now_iso(),
        }
        socketio.emit("chat_message", msg, to=room)

    # --- If chart/visualization data exists, emit separately ---
    if reply_data and "formatted_data" in reply_data:
        socketio.emit("chart_data", {"formatted_data": reply_data["formatted_data"]}, to=room)
