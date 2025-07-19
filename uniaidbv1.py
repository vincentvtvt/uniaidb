import os
import logging
import time
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
import requests
import openai
import base64
import re

# --- Logger setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("UniAI")

openai.api_key = os.getenv("OPENAI_API_KEY")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")

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

class Template(db.Model):
    __tablename__ = 'templates'
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.String(50))
    bot_id = db.Column(db.Integer)
    description = db.Column(db.String(255))
    content = db.Column(db.JSON)  # JSONB
    language = db.Column(db.String(10))
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)

# --- Helpers ---
def download_file(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def encode_image_b64(img_bytes):
    return base64.b64encode(img_bytes).decode()

def extract_text_from_image(img_url, prompt=None):
    image_bytes = download_file(img_url)
    img_b64 = encode_image_b64(image_bytes)
    logger.info("[VISION] Sending image to OpenAI Vision...")
    messages = [
        {"role": "system", "content": prompt or "Extract all visible text from this image. If no text, describe what you see."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
    ]
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
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
        img_url = msg.get("media", {}).get("url")
        if img_url:
            try:
                meaning = extract_text_from_image(
                    img_url,
                    prompt="This is a WhatsApp sticker. Describe the main emotion or meaning it conveys in 1-2 words, e.g. 'thank you', 'happy', 'love', 'celebration', 'laugh', etc. If text is visible, mention it."
                )
                return meaning or "[Sticker received]", img_url
            except Exception as e:
                logger.error(f"[STICKER MEANING] {e}")
                return "[Sticker received]", img_url
        return "[Sticker received]", None
    else:
        return f"[Unrecognized message type: {msg_type}]", None

def get_template_content(template_id):
    template = db.session.query(Template).filter_by(template_id=template_id, active=True).first()
    if not template or not template.content:
        return []
    return template.content if isinstance(template.content, list) else json.loads(template.content)

# --- Wassenger Send (Image/Template FIX) ---
def send_wassenger_reply(phone, text, device_id, delay_seconds=0, msg_type="text", caption=None):
    logger.info(f"[WASSENGER] To: {phone} | Device: {device_id} | Text: {text} | Type: {msg_type}")
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {
        "phone": phone,
        "device": device_id
    }
    if msg_type == "text":
        payload["message"] = text
    elif msg_type == "image":
        payload["mediaUrl"] = text
        if caption:
            payload["message"] = caption
        # If caption is empty, REMOVE 'message' to avoid validation error
        else:
            payload.pop("message", None)
    if delay_seconds > 0:
        deliver_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        payload["deliverAt"] = deliver_at.isoformat() + "Z"
        logger.info(f"[WASSENGER] Delayed send at: {payload['deliverAt']}")
    allowed_keys = {"phone", "device", "mediaUrl", "message", "deliverAt"}
    payload = {k: v for k, v in payload.items() if v is not None and k in allowed_keys}
    logger.debug(f"[WASSENGER PAYLOAD]: {payload}")
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

def get_bot_by_phone(phone_number):
    num_variants = [
        phone_number,
        phone_number.lstrip('+'),
        phone_number.replace('+', '').replace('@c.us', ''),
        '+' + phone_number.replace('@c.us', '').lstrip('+'),
        phone_number.replace('@c.us', ''),
    ]
    logger.debug(f"[DB] Bot lookup attempt for: {phone_number} (variants: {num_variants})")
    for variant in num_variants:
        bot = Bot.query.filter_by(phone_number=variant).first()
        if bot:
            logger.debug(f"[DB] Bot found! Input: {phone_number}, Matched: {variant} (DB: {bot.phone_number})")
            return bot
    logger.error(f"[DB] Bot NOT FOUND for: {phone_number} (tried: {num_variants})")
    return None

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
        max_tokens=1024,
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

def process_ai_reply_and_send(customer_phone, ai_reply, device_id, bot_id=None, user=None, session_id=None):
    try:
        parsed = ai_reply if isinstance(ai_reply, dict) else json.loads(ai_reply)
    except Exception:
        logger.error("[SEND SPLIT MSGS] Failed to parse AI reply as JSON")
        parsed = {}

    if "template" in parsed:
        template_id = parsed["template"]
        template_content = get_template_content(template_id)
        for part in template_content:
            if part.get("type") == "text":
                send_wassenger_reply(customer_phone, part["content"], device_id)
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", part["content"])
                time.sleep(1)
            elif part.get("type") == "image":
                # If part.get("caption") is empty, omit message field!
                send_wassenger_reply(
                    customer_phone,
                    part["content"],
                    device_id,
                    msg_type="image",
                    caption=part.get("caption") if part.get("caption") else None
                )
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", f"[IMAGE]{part['content']}")
                time.sleep(1)

    msg_lines = []
    if "message" in parsed:
        if isinstance(parsed["message"], list):
            msg_lines = parsed["message"]
        elif isinstance(parsed["message"], str):
            msg_lines = [parsed["message"]]
    elif isinstance(parsed, str):
        msg_lines = [parsed]
    for idx, part in enumerate(msg_lines[:3]):
        send_wassenger_reply(customer_phone, part, device_id)
        if bot_id and user and session_id:
            save_message(bot_id, user, session_id, "out", part)
        if idx < len(msg_lines[:3]) - 1:
            time.sleep(5)

@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    try:
        msg = data["data"]
        if msg.get("flow") == "outbound":
            return jsonify({"status": "ignored"}), 200

        bot_phone = msg.get("toNumber")
        user_phone = msg.get("fromNumber")
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

    process_ai_reply_and_send(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)

    if tool_id and "88" in tool_id:
        notify_sales_group(bot, f"Goal achieved for customer {user_phone}: {ai_reply}")

    return jsonify({"status": "ok", "ai_reply": ai_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
