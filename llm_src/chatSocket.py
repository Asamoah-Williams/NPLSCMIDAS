# llm_src/chatSocket.py
import os
import uuid

import yaml
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone, UTC
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from uuid import uuid4
from llm_src.utils.bot import bot_reply_async

# import eventlet  # noqa

# --- Config ---
ROOT = Path(__file__).resolve().parent / "configs"
CFG = yaml.safe_load((ROOT / "project_config.yml").read_text())

DEFAULT_ROOM = CFG["chat"]["default_room"]
MAX_HISTORY_PER_ROOM = CFG["chat"]["max_history"]
BOT_USERNAME = CFG["chat"]["bot_username"]
BOT_DELAY_SECONDS = float(CFG["chat"]["bot_delay_seconds"])

# --- State ---
sid_to_user = {}
room_users = defaultdict(set)
room_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_PER_ROOM))

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def emit_message_to_room(socketio, room: str, username: str, text: str, meta=None):
    """Helper to emit + persist a message."""
    msg = {
        "id": f"{username}-{datetime.now(UTC).timestamp()}",
        "username": username,
        "room": room,
        "text": text,
        "meta": meta or {},
        "ts": _now_iso(),
    }
    room_history[room].append(msg)
    socketio.emit("chat_message", msg, to=room)



def init_socketio(app):
    """Factory: create and attach SocketIO to Flask app"""

    # Try eventlet if available; fallback to threading (Windows safe)
    async_mode = os.getenv("ASYNC_MODE", None)

    if not async_mode:
        try:
            async_mode = "eventlet"
        except ImportError:
            async_mode = "threading"

    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="eventlet",
        logger=False,
        engineio_logger=False
    )

    # ---------------- SOCKET.IO HANDLERS ---------------- #

    @socketio.on("connect")
    def on_connect():
        emit("server_info", {"message": "connected", "rooms": list(room_users.keys())})


    @socketio.on("disconnect")
    def on_disconnect():
        info = sid_to_user.pop(request.sid, None)
        if info:
            username = info["username"]
            room = info["room"]
            if username in room_users[room]:
                room_users[room].remove(username)
                emit("user_left", {"username": username, "room": room, "ts": _now_iso()}, to=room)
                _push_user_list(room)

    def _push_user_list(room):
        """Send updated user list to everyone in the room"""
        users = list(room_users[room])
        emit("user_list", {"room": room, "users": users}, to=room)

    @socketio.on("join")
    def on_join(data):
        data = data or {}
        username = data.get("username")
        room = data.get("room")

        if not username or not isinstance(username, str):
            emit("error", {"error": "username required"})
            return

        # Generate a unique room per session if none provided
        if not room:
            room = f"room_{uuid.uuid4().hex}"

        # Leave all previous rooms except the default SID room
        for r in [r for r in rooms() if r != request.sid]:
            leave_room(r)

        join_room(room)

        # Store user session info
        session_id = f"s_{request.sid}"
        sid_to_user[request.sid] = {
            "username": username,
            "room": room,
            "session_id": session_id,
        }
        room_users[room].add(username)

        # Send message history for this room
        emit("history", {"room": room, "messages": list(room_history[room])})

        # Notify others in the room (excluding self)
        emit(
            "user_joined",
            {"username": username, "room": room, "ts": _now_iso()},
            to=room,
            include_self=False,
        )

        # Push updated user list
        _push_user_list(room)

        # Send the room UUID back to the client so it can reuse it if reconnecting
        emit("joined", {"room": room, "session_id": session_id})

    # @socketio.on("join")
    # def on_join(data):
    #     username = (data or {}).get("username")
    #     room = (data or {}).get("room") or DEFAULT_ROOM
    #
    #     if not username or not isinstance(username, str):
    #         emit("error", {"error": "username required"})
    #         return
    #
    #     for r in [r for r in rooms() if r != request.sid]:
    #         leave_room(r)
    #
    #     join_room(room)
    #
    #     session_id = f"s_{request.sid}"
    #     sid_to_user[request.sid] = {
    #         "username": username,
    #         "room": room,
    #         "session_id": session_id,
    #     }
    #     room_users[room].add(username)
    #
    #     emit("history", {"room": room, "messages": list(room_history[room])})
    #     emit("user_joined", {"username": username, "room": room, "ts": _now_iso()}, to=room, include_self=False)
    #     _push_user_list(room)
    #


    @socketio.on("send_message")
    def on_send_message(data):
        current = sid_to_user.get(request.sid)
        if not current:
            emit("error", {"error": "join a room first"})
            return

        text = (data or {}).get("text", "").strip()
        if not text:
            emit("error", {"error": "message text required"})
            return

        room = current["room"]

        emit_message_to_room(socketio, room, current["username"], text, meta=(data or {}).get("meta") or {})

        user_id = current["username"]
        session_id = current["session_id"]
        socketio.start_background_task(bot_reply_async, socketio, room, text, user_id, session_id,emit_message_to_room,)

    @socketio.on("typing")
    def on_typing(data):
        current = sid_to_user.get(request.sid)
        if not current:
            return
        room = (data or {}).get("room") or current["room"]
        is_typing = bool((data or {}).get("is_typing", True))
        emit(
            "typing",
            {"username": current["username"], "room": room, "is_typing": is_typing, "ts": _now_iso()},
            to=room,
            include_self=False,
        )

    def _push_user_list(room):
        socketio.emit("user_list", {"room": room, "users": sorted(list(room_users[room]))}, to=room)

    return socketio
