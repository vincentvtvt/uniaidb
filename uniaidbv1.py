import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime, timedelta
import requests
import openai
import base64
import re
import json

openai.api_key = os.getenv("OPENAI_API_KEY")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAI")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Models ---
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
    customer_phone = db.Column(db.String(50))
    session_id = db.Column(db.String(50))
    direction = db.Column(db.String(10))  # 'in' or 'out'
    content = db.Column(db.Text)
    raw_media_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime)

# --- Media/Text Extraction ---
def download_file(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def encode_image_b64(img_bytes):
    return base64.b64encode(img_bytes).decode()

def extract_text_from_image(img_url):
    image_bytes = download_file(img_url)
    img_b64 = encode_image_b64(image_bytes)
    logger.info("[VISION] Sending image to OpenAI Vision...")
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract all visible text from this image. If no text, describe what you see."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
        ],
        max_tokens=256
    )
    return result.choices[0].message.content.strip()

def transcribe_audio_from_url(audio_url):
    audio_bytes = download_file(audio_url)
    temp_path = "/tmp/temp_audio.ogg"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    with open(temp_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text.strip()

def extract_text_from_message(msg):
    msg_type = msg.get("type", "text")
    logger.info(f"[MEDIA DETECT] Message type: {msg_type}")
    if msg_type == "text":
        return msg.get("body", ""), None
    elif msg_type == "image":
        img_url = msg.get("media", {}).get("url")
        caption = msg.get("body", "")
        try:
            ocr_text = extract_text_from_image(img_url) if img_url else ""
        except Exception as e:
            logger.error(f"[IMAGE OCR] {e}")
            ocr_text = ""
        combined = " ".join(filter(None, [caption, ocr_text]))
        return combined.strip() or "[Image received, no text found]", img_url
    elif msg_type == "audio":
        audio_url = msg.get("media", {}).get("url")
        try:
            transcript = transcribe_audio_from_url(audio_url) if audio_url else "[Audio received, no url]"
            return transcript, audio_url
        except Exception as e:
            logger.error(f"[AUDIO TRANSCRIBE] {e}")
            return "[Audio received, transcription failed]", audio_url
    elif msg_type == "sticker":
        return "[Sticker received]", None
    else:
        return f"[Unrecognized message type: {msg_type}]", None

# --- Wassenger Send Message (with delayed delivery via API, support for images and templates) ---
def send_wassenger_reply(phone, payload, device_id, delay_seconds=0, msg_type="text"):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}

    data = {"device": device_id, "phone": phone}
    deliver_at = None
    if delay_seconds > 0:
        deliver_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        data["deliverAt"] = deliver_at.isoformat() + "Z"

    if msg_type == "image" and isinstance(payload, dict):
        data["media"] = {"url": payload["url"]}
        if payload.get("caption"):
            data["message"] = payload["caption"]
    else:
        data["message"] = payload

    logger.info(f"[WASSENGER] To: {phone} | Type: {msg_type} | Device: {device_id} | Data: {data}")
    try:
        resp = requests.post(url, json=data, headers=headers)
        logger.info(f"Wassenger response: {resp.text}")
    except Exception as e:
        logger.error(f"Wassenger send failed: {e}")

def notify_sales_group(bot, message, error=False):
    group_id = (bot.config or {}).get("notification_group")
    device_id = (bot.config or {}).get("device_id")
    if group_id and device_id:
        note = f"[ALERT] {message}" if error else message
        send_wassenger_reply(group_id, note, device_id)
    else:
        logger.warning("[NOTIFY] Notification group or device_id missing in bot.config")

# --- DB Utility ---
def get_bot_by_phone(phone_number):
    bot = Bot.query.filter_by(phone_number=phone_number).first()
    logger.info(f"[DB] Bot lookup by phone: {phone_number} => {bot}")
    return bot

def get_active_tools_for_bot(bot_id):
    tools = (
        db.session.query(Tool)
        .join(BotTool, and_(
            Tool.tool_id == BotTool.tool_id,
            BotTool.bot_id == bot_id,
            Tool.active == True,
            BotTool.active == True
        )).all()
    )
    logger.info(f"[DB] Tools for bot_id={bot_id}: {[t.tool_id for t in tools]}")
    return tools

def save_message(bot_id, customer_phone, session_id, direction, content, raw_media_url=None):
    msg = Message(
        bot_id=bot_id,
        customer_phone=customer_phone,
        session_id=session_id,
        direction=direction,
        content=content,
        raw_media_url=raw_media_url,
        created_at=datetime.now()
    )
    db.session.add(msg)
    db.session.commit()
    logger.info(f"[DB] Saved message ({direction}) for {customer_phone}: {content}")

def get_latest_history(bot_id, customer_phone, session_id, n=20):
    messages = (Message.query
        .filter_by(bot_id=bot_id, customer_phone=customer_phone, session_id=session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
        .all())
    messages = messages[::-1]
    logger.info(f"[DB] History ({len(messages)} messages) loaded.")
    return messages

# --- AI: Tool Decision ---
def decide_tool_with_manager_prompt(bot, history):
    prompt = bot.manager_system_prompt
    history_text = "\n".join([f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}" for m in history])
    logger.info(f"[AI DECISION] manager_system_prompt: {prompt}")
    logger.info(f"[AI DECISION] history: {history_text}")
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history_text}
        ],
        max_tokens=32,
        temperature=0
    )
    tool_decision = response.choices[0].message.content
    logger.info(f"[AI DECISION] Tool chosen: {tool_decision}")
    match = re.search(r'"TOOLS":\s*"([^"]+)"', tool_decision)
    return match.group(1) if match else None

# --- AI: Final Reply Generator (Streaming) ---
def compose_reply(bot, tool, history, context_input):
    if tool:
        prompt = (bot.system_prompt or "") + "\n" + (tool.prompt or "")
    else:
        prompt = bot.system_prompt or ""
    logger.info(f"[AI REPLY] Prompt: {prompt}")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context_input}
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=8192,
        temperature=0.3,
        stream=True
    )
    reply_accum = ""
    print("[STREAM] Streaming model reply:")
    for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            reply_accum += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
    logger.info(f"\n[AI REPLY STREAMED]: {reply_accum}")
    return reply_accum

# --- Split and Send Multi-part (template-aware, delayed, correct order) ---
def send_split_messages_wassenger(phone, ai_reply, device_id, bot_id=None, user=None, session_id=None):
    try:
        arr = json.loads(ai_reply)
        # If AI reply is a list (template): [{"type": "text", ...}, {"type": "image", ...}]
        if isinstance(arr, list):
            for idx, item in enumerate(arr):
                if item["type"] == "text":
                    send_wassenger_reply(phone, item["content"], device_id, delay_seconds=5*idx, msg_type="text")
                    if bot_id and user and session_id:
                        save_message(bot_id, user, session_id, "out", item["content"])
                elif item["type"] == "image":
                    send_wassenger_reply(phone, {"url": item["content"]}, device_id, delay_seconds=5*idx, msg_type="image")
                    if bot_id and user and session_id:
                        save_message(bot_id, user, session_id, "out", f"[image] {item['content']}")
            return
        # If AI reply is dict {"message": [...]}, follow the array
        elif isinstance(arr, dict) and "message" in arr and isinstance(arr["message"], list):
            for idx, msg in enumerate(arr["message"]):
                send_wassenger_reply(phone, msg, device_id, delay_seconds=5*idx, msg_type="text")
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", msg)
            return
    except Exception:
        pass  # Not JSON or not in correct format

    # Otherwise, split by line or chunk if too long (fallback)
    max_len = 1024
    parts = []
    if "\n" in ai_reply and len(ai_reply) < max_len:
        parts = [p.strip() for p in ai_reply.split("\n") if p.strip()]
    else:
        parts = [ai_reply[i:i+max_len] for i in range(0, len(ai_reply), max_len)]
    for idx, part in enumerate(parts):
        send_wassenger_reply(phone, part, device_id, delay_seconds=5*idx, msg_type="text")
        if bot_id and user and session_id:
            save_message(bot_id, user, session_id, "out", part)

# --- Webhook Handler ---
@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    try:
        msg = data["data"]
        event_type = data.get("event", "")
        # Always use toNumber as bot's own number for inbound, fromNumber for outbound (for bot lookup)
        if event_type.startswith("message:in"):
            bot_phone = msg.get("toNumber")
            user_phone = msg.get("fromNumber")
        else:  # message:out, update, etc.
            bot_phone = msg.get("fromNumber")
            user_phone = msg.get("toNumber")
        device_id = data["device"]["id"]
        session_id = user_phone
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    msg_text, raw_media_url = extract_text_from_message(msg)

    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404

    if not msg_text or msg_text.startswith("[Unrecognized") or msg_text.startswith("[Audio received, transcription failed]"):
        logger.error("[ERROR] Failed to extract text from message.")
        notify_sales_group(bot, f"Failed to process customer message: {raw_media_url}", error=True)
        return jsonify({"error": "Failed to process customer message"}), 500

    save_message(bot.id, user_phone, session_id, "in", msg_text, raw_media_url=raw_media_url)

    history = get_latest_history(bot.id, user_phone, session_id)

    tool_id = decide_tool_with_manager_prompt(bot, history)
    tool = None
    if tool_id and tool_id.lower() != "default":
        for t in get_active_tools_for_bot(bot.id):
            if t.tool_id == tool_id:
                tool = t
                break
    logger.info(f"[LOGIC] Tool selected: {tool_id}, tool obj: {tool}")

    context_input = (
        "\n".join([f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}" for m in history])
        if tool else msg_text
    )

    ai_reply = compose_reply(bot, tool, history, context_input)
    send_split_messages_wassenger(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)

    if "success" in ai_reply.lower() or "booking confirmed" in ai_reply.lower():
        notify_sales_group(bot, f"Goal achieved for customer {user_phone}: {ai_reply}")

    return jsonify({"status": "ok", "ai_reply": ai_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
