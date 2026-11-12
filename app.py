"""Asnivo - Flask backend: POST /api/chat, POST /api/reset, GET /api/health + static UI."""
import os

from flask import Flask, jsonify, request, send_from_directory

from chatbot import ChatBot, ChatError, ConversationStore

app = Flask(__name__, static_folder="static", static_url_path="/static")
bot = ChatBot()
store = ConversationStore()
MAX_MESSAGE_CHARS = 2000


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Please type a message first."), 400
    message = message.strip()
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"Message is too long (max {MAX_MESSAGE_CHARS} characters)."), 400

    requested_id = data.get("conversation_id")
    if requested_id is not None and not isinstance(requested_id, str):
        return jsonify(error="conversation_id must be a string."), 400
    cid, conv = store.get_or_create(requested_id)
    try:
        reply = bot.reply(conv, message)
    except ChatError as err:
        return jsonify(error=err.user_message, conversation_id=cid), err.status
    return jsonify(
        reply=reply,
        conversation_id=cid,
        stored_messages=len(conv.messages),
        has_summary=bool(conv.summary),
    )


@app.post("/api/reset")
def reset():
    data = request.get_json(silent=True)
    if isinstance(data, dict) and isinstance(data.get("conversation_id"), str):
        store.delete(data["conversation_id"])
    return jsonify(status="reset")


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=5000)
