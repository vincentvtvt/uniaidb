import os
import logging
import time
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime
import requests

# OpenAI/Anthropic imports (stubs shown, fill in your own!)
import openai
# from anthropic import Anthropic

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAI")

# === Flask/SQLAlchemy Setup ===
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# === Models ===
class Bot(db.Model):
    __tablename__ = 'bots'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    phone_number = db.Column(db.String(30))
    config = db.Column(db.JSON)
    created_at = db.Column(db.DateTime)
    system_prompt = db.Column(db.Text)
    manager_system_prompt = db.Column(db.Text)

class Tool(db.Model):
    __tablename__ = 'tools'
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.String(50))
    name = db.Column(db.String(100))
    description = db.Column(db.Text)
    prompt = db.Column(db.Text)
    action_type = db.Column(db.String(30))
    active = db.Column(db.Boolean)
    created_at = db.Column(db.DateTime)

class BotTool(db.Model):
    __tablename__ = 'bot_tools'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    tool_id = db.Column(db.String(50))
    active = db.Column(db.Boolean)

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    customer_phone = db.Column(db.String(30))
    session_id = db.Column(db.String(50))
    direction = db.Column(db.String(10))  # 'in' or 'out'
    content = db.Column(db.Text)
    raw_media_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime)

# === Wassenger WhatsApp Send (Live) ===
def send_wassenger_reply(phone, text, device_id):
    logger.info(f"[WASSENGER] To: {phone} | Device: {device_id} | Text: {text}")
    url = "https://api.wassenger.com/v1/messages"
    headers = {
        "Token": os.getenv("WASSENGER_API_KEY"),
        "Content-Type": "application/json"
    }
    payload = {"phone": phone, "message": text, "device": device_id}
    resp = requests.post(url, json=payload, headers=headers)
    logger.info(f"Wassenger response: {resp.text}")

# === AI: Media to Text Conversion (Stub) ===
def ai_extract_text(msg):
    if msg["type"] == "image":
        # Use OpenAI Vision API (or similar) to get caption/text
        image_url = msg["mediaUrl"]
        logger.info(f"[MEDIA] Vision extracting from: {image_url}")
        # vision_resp = openai.Image.create(...); simulated here:
        return "[Image] (text content extracted by Vision API)"
    elif msg["type"] == "audio":
        audio_url = msg["mediaUrl"]
        logger.info(f"[MEDIA] Whisper extracting from: {audio_url}")
        # whisper_resp = openai.Audio.transcribe(...); simulated here:
        return "[Audio] (transcribed by Whisper API)"
    elif msg["type"] == "sticker":
        return "[Sticker received]"
    else:
        return "[Unsupported media type]"

# === DB: Bot by phone ===
def get_bot_by_phone(phone_number):
    bot = Bot.query.filter_by(phone_number=phone_number).first()
    logger.info(f"[DB] Bot lookup by phone: {phone_number} => {bot}")
    return bot

# === DB: Active Tools for Bot ===
def get_active_tools_for_bot(bot_id):
    tools = (
        db.session.query(Tool)
        .join(BotTool, and_(
            Tool.tool_id == BotTool.tool_id,
            BotTool.bot_id == bot_id,
            Tool.active == True,
            BotTool.active == True
        ))
        .all()
    )
    logger.info(f"[DB] Tools for bot_id={bot_id}: {[t.tool_id for t in tools]}")
    return tools

def get_tool_by_id(tools, tool_id):
    for t in tools:
        if t.tool_id == tool_id:
            return t
    return None

# === DB: Message Save & Get History ===
def save_message(bot_id, customer_phone, session_id, direction, content, raw_media_url=None):
    msg = Message(
        bot_id=bot_id, customer_phone=customer_phone, session_id=session_id,
        direction=direction, content=content, raw_media_url=raw_media_url, created_at=datetime.utcnow())
    db.session.add(msg)
    db.session.commit()
    logger.info(f"[DB] Message saved: {direction}, {content}")

def get_recent_history(bot_id, customer_phone, session_id, max_msgs=20):
    messages = Message.query.filter_by(
        bot_id=bot_id,
        customer_phone=customer_phone,
        session_id=session_id
    ).order_by(Message.created_at.desc()).limit(max_msgs).all()
    history = []
    for m in reversed(messages):
        role = "user" if m.direction == "in" else "assistant"
        history.append({"role": role, "content": m.content})
    logger.info(f"[DB] Fetched history: {len(history)} messages")
    return history

# === AI: Tool Selection ===
def ai_select_tool(manager_prompt, history):
    # Build the prompt for tool selection (manager_system_prompt + history)
    prompt = manager_prompt + "\n"
    for h in history[-10:]:
        prompt += f"{h['role']}: {h['content']}\n"
    # TODO: swap with your actual Claude or GPT call for tool selection!
    logger.info(f"[AI] Tool selector prompt: {prompt}")
    # For now: always default (replace with real LLM)
    return "defaultvpt"

# === AI: Generate Reply ===
def ai_generate_reply(system_prompt, tool_prompt, history, user_message):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if tool_prompt:
        messages.append({"role": "system", "content": tool_prompt})
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_message})

    logger.info(f"[AI] Final messages to Claude/GPT: {messages}")

    # Stream reply (stub with OpenAI; replace model with Claude if needed)
    reply = ""
    for token in ["[simulated", "Claude/GPT", "response", "here]"]:
        time.sleep(0.1)
        print(token, end=' ', flush=True)
        reply += token + " "
    print()
    return reply.strip()

def smart_split(text, max_parts=3):
    # Simple split for demo: split by '.', max_parts
    parts = text.split('. ')
    if len(parts) <= max_parts:
        return [p.strip() for p in parts if p.strip()]
    avg = len(parts) // max_parts
    out = []
    for i in range(max_parts):
        out.append('. '.join(parts[i*avg:(i+1)*avg]).strip())
    return [o for o in out if o]

# === Main Webhook ===
@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    # Wassenger extraction
    try:
        msg = data["data"]
        bot_phone = msg["toNumber"]
        user_phone = msg["fromNumber"]
        device_id = data["device"]["id"]
        msg_type = msg["type"]
        msg_text = msg.get("body", "")
        session_id = user_phone  # session logic as needed
        raw_media_url = msg.get("mediaUrl", "")
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # Bot lookup
    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404

    # Step 2: Save message (with media-to-text)
    if msg_type == "text":
        save_message(bot.id, user_phone, session_id, "in", msg_text)
        user_message = msg_text
    else:
        user_message = ai_extract_text(msg)
        save_message(bot.id, user_phone, session_id, "in", user_message, raw_media_url=raw_media_url)

    # Step 3: Get tools and history
    tools = get_active_tools_for_bot(bot.id)
    history = get_recent_history(bot.id, user_phone, session_id)

    # Step 4: AI tool selection
    tool_id = ai_select_tool(bot.manager_system_prompt, history)
    tool = get_tool_by_id(tools, tool_id)

    # Step 5: Build prompt for AI reply
    system_prompt = bot.system_prompt or ""
    tool_prompt = tool.prompt if tool else ""
    final_prompt = f"{system_prompt}\n{tool_prompt}"

    logger.info(f"[PROMPT] System: {system_prompt[:50]}... Tool: {tool_prompt[:50]}...")

    # Step 6: AI generates reply (stream, print progress)
    ai_reply = ai_generate_reply(system_prompt, tool_prompt, history, user_message)

    # Step 7: Split reply, send via Wassenger, save out
    parts = smart_split(ai_reply, max_parts=3)
    for part in parts:
        send_wassenger_reply(user_phone, part, device_id)
        time.sleep(1)
        save_message(bot.id, user_phone, session_id, "out", part)

    return jsonify({"status": "ok", "tool_used": tool_id, "reply": ai_reply})

# === Start Flask app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
