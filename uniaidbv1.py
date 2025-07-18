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

# --- SQLAlchemy Models (same as before, with raw_media_url field for Message) ---
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
    raw_media_url = db.Column(db.Text)   # Only for non-text
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

# --- Wassenger Send Message (real) ---
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

def notify_sales_group(bot, message, error=False):
    group_id = (bot.config or {}).get("notification_group")
    device_id = (bot.config or {}).get("device_id")
    if group_id and device_id:
        note = f"[ALERT] {message}" if error else message
        send_wassenger_reply(group_id, note, device_id)
    else:
        logger.warning("[NOTIFY] Notification group or device_id missing in bot.config")

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

# --- Claude/AI Tool Decision (Step 1) ---
def decide_tool_with_manager_prompt(bot, history):
    prompt = bot.manager_system_prompt
    history_text = "\n".join(
        [f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}" for m in history]
    )
    logger.info(f"[AI DECISION] manager_system_prompt: {prompt}")
    logger.info(f"[AI DECISION] history for decision: {history_text}")
    # Call Claude/GPT here to select tool_id
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history_text}
        ],
        max_tokens=32,
        temperature=0
    )
    tool_decision = response["choices"][0]["message"]["content"]
    logger.info(f"[AI DECISION] Tool chosen: {tool_decision}")
    # Extract tool_id from response, fallback to None if parsing fails
    import re
    match = re.search(r'"TOOLS":\s*"([^"]+)"', tool_decision)
    return match.group(1) if match else None

# --- Claude/AI Final Reply Generator (Step 2/3) ---
def compose_reply(bot, tool, history, context_input):
    if tool:
        prompt = (bot.system_prompt or "") + "\n" + (tool.prompt or "")
    else:
        prompt = bot.system_prompt or ""
    logger.info(f"[AI REPLY] Prompt to model: {prompt}")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context_input}
    ]
    # Stream reply (simulate, you can do OpenAI stream if needed)
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # replace with claude-3-7-sonnet-20250219 as needed
        messages=messages,
        max_tokens=512,
        temperature=0.3
    )
    ai_reply = response["choices"][0]["message"]["content"].strip()
    logger.info(f"[AI REPLY] {ai_reply}")
    return ai_reply

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
        session_id = user_phone  # or your session policy
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # --- Extract message as text (universal handler) ---
    msg_text, raw_media_url = extract_text_from_message(msg)

    # Save IN message only if text is available, otherwise notify group
    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404

    if not msg_text or msg_text.startswith("[Unrecognized") or msg_text.startswith("[Audio received, transcription failed]"):
        logger.error("[ERROR] Failed to extract text from message.")
        notify_sales_group(bot, f"Failed to process customer message: {raw_media_url}", error=True)
        return jsonify({"error": "Failed to process customer message"}), 500

    save_message(bot.id, user_phone, session_id, "in", msg_text, raw_media_url=raw_media_url)

    # Get latest history
    history = get_latest_history(bot.id, user_phone, session_id)

    # Step 1: Decide tool using manager_system_prompt
    tool_id = decide_tool_with_manager_prompt(bot, history)
    tool = None
    if tool_id and tool_id != "Default":
        tools = get_active_tools_for_bot(bot.id)
        for t in tools:
            if t.tool_id == tool_id:
                tool = t
                break
    logger.info(f"[LOGIC] Tool selected: {tool_id}, tool obj: {tool}")

    # Step 2: Use tool prompt/context (if any), else just history as input
    if tool:
        context_input = "\n".join([f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}" for m in history])
    else:
        context_input = msg_text

    # Step 3: Compose AI reply (Claude/GPT)
    ai_reply = compose_reply(bot, tool, history, context_input)

    # Step 4: Send AI reply (split as needed)
    max_len = 512  # or per your requirement
    reply_chunks = [ai_reply[i:i+max_len] for i in range(0, len(ai_reply), max_len)]
    for chunk in reply_chunks:
        send_wassenger_reply(user_phone, chunk, device_id)
        save_message(bot.id, user_phone, session_id, "out", chunk)

    # Step 5: If goal achieved (define your goal logic), notify group
    if "success" in ai_reply.lower() or "booking confirmed" in ai_reply.lower():
        notify_sales_group(bot, f"Goal achieved for customer {user_phone}: {ai_reply}")

    return jsonify({"status": "ok", "ai_reply": ai_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
