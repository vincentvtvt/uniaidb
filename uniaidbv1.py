import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime
import requests
import openai
import base64

# --- Configuration ---
openai.api_key = os.getenv("OPENAI_API_KEY")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY", "your-wassenger-api-key")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAI")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- SQLAlchemy Models ---
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
    direction = db.Column(db.String(10)) # 'in' or 'out'
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime)

# --- Universal Media/Text Extractor ---
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
    result = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract all visible text from this image. If no text, describe what you see."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}
        ],
        max_tokens=256
    )
    vision_result = result["choices"][0]["message"]["content"].strip()
    logger.info(f"[VISION RESULT] {vision_result}")
    return vision_result

def transcribe_audio_from_url(audio_url):
    audio_bytes = download_file(audio_url)
    temp_path = "/tmp/temp_audio.ogg"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    with open(temp_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    transcript_text = transcript["text"].strip()
    logger.info(f"[WHISPER] Transcription: {transcript_text}")
    return transcript_text

def extract_text_from_message(msg):
    msg_type = msg.get("type", "text")
    logger.info(f"[MEDIA DETECT] Message type: {msg_type}")
    if msg_type == "text":
        return msg.get("body", "")
    elif msg_type == "image":
        img_url = msg.get("media", {}).get("url")
        caption = msg.get("body", "")
        try:
            ocr_text = extract_text_from_image(img_url) if img_url else ""
        except Exception as e:
            logger.error(f"[IMAGE OCR] {e}")
            ocr_text = ""
        combined = " ".join(filter(None, [caption, ocr_text]))
        return combined.strip() or "[Image received, no text found]"
    elif msg_type == "audio":
        audio_url = msg.get("media", {}).get("url")
        try:
            return transcribe_audio_from_url(audio_url) if audio_url else "[Audio received, no url]"
        except Exception as e:
            logger.error(f"[AUDIO TRANSCRIBE] {e}")
            return "[Audio received, transcription failed]"
    elif msg_type == "sticker":
        return "[Sticker received]"
    else:
        return f"[Unrecognized message type: {msg_type}]"

# --- Wassenger Send Message (Stub) ---
def send_wassenger_reply(phone, text, device_id):
    logger.info(f"[WASSENGER] To: {phone} | Device: {device_id} | Text: {text}")
    url = "https://api.wassenger.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "Token": WASSENGER_API_KEY
    }
    payload = {
        "phone": phone,
        "message": text,
        "device": device_id
    }
    try:
        resp = requests.post(url, json=payload, headers=headers)
        logger.info(f"Wassenger response: {resp.text}")
    except Exception as e:
        logger.error(f"Wassenger send failed: {e}")

# --- DB Logic (minimal) ---
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
        ))
        .all()
    )
    logger.info(f"[DB] Tools for bot_id={bot_id}: {[t.tool_id for t in tools]}")
    return tools

def find_default_tool(tools):
    for tool in tools:
        if tool.tool_id == "defaultvpt":
            return tool
    return None

def save_message(bot_id, customer_phone, session_id, direction, content):
    msg = Message(
        bot_id=bot_id,
        customer_phone=customer_phone,
        session_id=session_id,
        direction=direction,
        content=content,
        created_at=datetime.now()
    )
    db.session.add(msg)
    db.session.commit()
    logger.info(f"[DB] Saved message ({direction}) for {customer_phone}: {content}")

# --- Webhook Handler ---
@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    try:
        msg = data["data"]
        bot_phone = msg["toNumber"]
        user_phone = msg["fromNumber"]
        device_id = data["device"]["id"]
        session_id = user_phone  # or define another session policy
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # --- Extract message as text (universal handler) ---
    msg_text = extract_text_from_message(msg)

    # Save IN message
    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404
    save_message(bot.id, user_phone, session_id, "in", msg_text)

    # --- Tool logic / prompt selection ---
    tools = get_active_tools_for_bot(bot.id)
    default_tool = find_default_tool(tools)
    if default_tool:
        final_prompt = (bot.system_prompt or "") + "\n" + (default_tool.prompt or "")
        logger.info(f"[PROMPT] Using system_prompt + defaultvpt: {final_prompt}")
    else:
        final_prompt = bot.system_prompt or ""
        logger.warning("[PROMPT] No defaultvpt tool found, using system_prompt only")
    logger.info(f"[PROMPT] Final prompt: {final_prompt}")

    # --- Send reply via Wassenger ---
    send_wassenger_reply(user_phone, final_prompt, device_id)

    # Save OUT message
    save_message(bot.id, user_phone, session_id, "out", final_prompt)

    return jsonify({"status": "ok", "used_prompt": final_prompt})

# --- Flask run (dev only) ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
