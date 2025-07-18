import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime
import requests
from anthropic import Anthropic, AsyncAnthropic

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAI")

# --- Flask + DB ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Claude config ---
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("LAUDE_MODEL", "claude-3-7-sonnet-20250219")  # typo as in your variable, fix as needed

# --- Models ---
class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer)
    sender_type = db.Column(db.String(20))
    sender_id = db.Column(db.Integer)
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta = db.Column(db.JSON)
    payload = db.Column(db.JSON)

# Add your other tables/models as needed for tool, bot, etc.

# --- Utility ---
def log_message(session_id, sender_type, sender_id, message, meta=None, payload=None):
    msg = Message(
        session_id=session_id,
        sender_type=sender_type,
        sender_id=sender_id,
        message=message,
        created_at=datetime.utcnow(),
        meta=meta or {},
        payload=payload or {}
    )
    db.session.add(msg)
    db.session.commit()
    logger.info(f"[LOG] Message saved: session_id={session_id}, sender_type={sender_type}, sender_id={sender_id}, message='{message}'")
    return msg.id

def send_wassenger_reply(phone, text, device_id):
    logger.info(f"[WASSENGER] To: {phone} | Device: {device_id} | Text: {text}")
    # Example only; fill in your actual logic.
    # url = "https://api.wassenger.com/v1/messages"
    # headers = {"Token": os.getenv("WASSENGER_API_KEY")}
    # payload = {"phone": phone, "message": text, "device": device_id}
    # resp = requests.post(url, json=payload, headers=headers)
    # logger.info(f"Wassenger response: {resp.text}")

# --- Claude Streaming Reply ---
def get_claude_stream(system_prompt, user_message, history):
    import openai
    openai.api_key = CLAUDE_API_KEY
    # Compose message history for Claude
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # add history
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    # add current message
    messages.append({"role": "user", "content": user_message})
    # NOTE: For Anthropic, you should use anthropic.Message.create for streaming, below is sample logic.
    from anthropic import Anthropic, AsyncAnthropic
    client = Anthropic(api_key=CLAUDE_API_KEY)
    stream = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.3,
        stream=True
    )
    full_reply = ""
    for event in stream:
        if event.type == "content_block_delta":
            content = event.delta.text
            print(content, end="", flush=True)  # streaming log
            full_reply += content
    print()  # new line after stream
    return full_reply

# --- Webhook ---
@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    # Parse Wassenger webhook
    try:
        msg = data["data"]
        bot_phone = msg["toNumber"]
        user_phone = msg["fromNumber"]
        device_id = data["device"]["id"]
        msg_text = msg["body"]
        # Session logic here, for now use user_phone as session_id
        session_id = int(''.join(filter(str.isdigit, user_phone)))  # naive example, use your session table in prod
        user_id = session_id
        bot_id = 1
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # 1. Log incoming message
    log_message(session_id=session_id, sender_type="user", sender_id=user_id, message=msg_text, meta=None, payload=data)

    # 2. Retrieve message history for session (limit as needed)
    messages = (
        Message.query.filter_by(session_id=session_id)
        .order_by(Message.created_at.asc())
        .limit(20).all()
    )
    claude_history = [
        {
            "role": "user" if m.sender_type == "user" else "assistant",
            "content": m.message
        }
        for m in messages
    ]

    # 3. Prepare prompts (stub, pull from DB as needed)
    system_prompt = "You are a helpful WhatsApp bot."  # replace with your bot table system_prompt
    tools_prompt = ""  # logic for tool prompt selection here

    # 4. Claude streaming reply
    try:
        bot_reply = get_claude_stream(system_prompt, msg_text, claude_history)
    except Exception as ex:
        logger.error(f"[CLAUDE] Error: {ex}")
        bot_reply = "Sorry, I can't process your request right now."

    # 5. Log bot reply
    log_message(session_id=session_id, sender_type="bot", sender_id=bot_id, message=bot_reply, meta=None, payload=None)

    # 6. Send reply via Wassenger
    send_wassenger_reply(user_phone, bot_reply, device_id)

    return jsonify({"status": "ok"})

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
