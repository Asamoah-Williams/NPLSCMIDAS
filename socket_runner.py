import eventlet
eventlet.monkey_patch()
from flask import Flask
from flask_cors import CORS
from llm_src.chatSocket import init_socketio


app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    # Waitress running port: 5001
    # Socket io running port : 5002
    socketio = init_socketio(app)
    socketio.run(app, host='127.0.0.1', port=5002)