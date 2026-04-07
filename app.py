import json
from flask import Flask, jsonify, request
from pathlib import Path
import pandas as pd
import numpy as np
import sys
from flask_cors import CORS
from npl_src.db import DatabaseConnection
from npl_src.db_extraction import extract
# from llm_src.chatSocket import init_socketio
import os
from npl_src.processor import processor
from npl_src.procedures import execute_procedures, list_procedures
from npl_src.forecast import forecast
import threading

app = Flask(__name__)
CORS(app)

# ---------- Reporting hooks (optional backtest summary) -----------------------

db_con = DatabaseConnection()

# ---------- Repo roots & import path ------------------------------------------
# Set this file NEXT TO (or inside the same src/ tree as) processor.py.
# If this file lives at project root and processor.py is under src/,
# this finds and adds src/ to sys.path.
ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "src") if (ROOT / "src").exists() else ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # ensure "src" modules import cleanly


# Import the pipeline entrypoint (note: it's __entry_main, not _entry_main)


@app.route('/training', methods=['GET'])
def get_train():
    try:
        # This will run ythe full workflow (DQ -> transform -> train -> backtest -> monitoring)
        # obj = processor()
        # obj.run_main()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        # Surface any pipeline errors as JSON
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/procedures", methods=["POST"])
def procedures_handler():
    data = request.get_json()

    if not data or "action" not in data:
        return jsonify({"error": "Invalid request"}), 400

    action = data["action"]

    # 🔹 PAGE LOAD
    if action == "list":
        return jsonify({"procedures": list_procedures()})

    # 🔹 BUTTON CLICK
    if action == "execute":
        procedure_ids = data.get("procedures", [])

        if not procedure_ids:
            return jsonify({"error": "No procedures selected"}), 400

        if action == "execute":
            return jsonify(execute_procedures(procedure_ids))

    return jsonify({"error": "Unsupported action"}), 400


@app.route('/train_model', methods=['GET'])
def get_train_model():
    try:
        def run_training_task(app_context):
            with app_context:
                print("--- Training Started in Background ---")
                obj = processor()
                obj.run_main()
                print("--- Training Completed ---")

        thread = threading.Thread(target=run_training_task, args=(app.app_context(),))
        thread.start()

        return jsonify({
            "status": "accepted",
            "message": "Model training has started in the background."
        }), 202

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/testing', methods=['GET'])
def get_test_model():
    try:
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/forecast', methods=['GET'])
def get_forecast():
    try:
        # This will run forecast
        obj = forecast()
        obj.run_main()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        # Surface any pipeline errors as JSON
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/dbextract', methods=['GET'])
def extraction():
    obj = extract()
    tables = obj.db_t()

    json_tables = {}
    for table_name, df in tables.items():
        # Replace +/- inf with NA, then let pandas handle NaN→null and datetimes→ISO
        safe_df = df.replace([np.inf, -np.inf], pd.NA)

        # pandas will output proper JSON nulls; parse back to Python for jsonify()
        records = json.loads(
            safe_df.to_json(orient='records', date_format='iso')
        )
        json_tables[table_name] = records

    return jsonify(json_tables)


if __name__ == "__main__":
    try:
        # with app.app_context():
        #     print("Testing model training end point:...............")
        #     get_train()
        #     print("End of model training:.....................")

        print("About to start server...")
        # socketio = init_socketio(app)
        port = int(os.getenv("PORT", 5001))
        print(f"🚀 Server running on http://localhost:{port}")

        app.run(
            host="127.0.0.1",
            port=port,
            debug=True
        )
        # socketio.run(
        #     app,
        #     host="0.0.0.0",
        #     port=port,
        #     debug=False,
        #     use_reloader=False,
        #     allow_unsafe_werkzeug=True,
        # )
    except Exception as ex:
        print("Sorry... start up error: " + str(ex))
