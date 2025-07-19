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
import time

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

class Template(db.Model):
    __tablename__ = 'templates'
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.String(100), unique=True)
    content = db.Column(db.Text)  # JSON: [{"type":"text","content":...},...]

# --- Phone Formatter ---
def format_e164(phone):
    phone = str(phone).replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
    if phone.startswith('+'):
        return phone
    if phone.startswith('6'):
        return '+' + phone
    if phone.startswith('01'):
        return '+6' + phone
    if phone.startswith('1'):
        return '+60' + phone
    return phone

def get_or_create_customer(phone):
    # Optionally: create or fetch from customer table
    return {"phone": phone}

# --- AI Interpreter for Any Media ---
def ai_interpret_file(file_url, file_type):
    try:
        resp = requests.get(file_url)
        resp.raise_for_status()
        file_bytes = resp.content
        if file_type in ("image", "sticker", "video"):
            encoded = base64.b64encode(file_bytes).decode()
            msg = (
                "You are an AI WhatsApp interpreter. User sent a "
                f"{file_type}. If it's a sticker, briefly describe its meaning in one sentence "
                "(e.g. 'User sent sticker of a cat laughing'). For image/video, describe the visual, "
                "OCR any visible text, or summarize content. Reply always in the main language used by the customer. "
                "If the last message was Chinese, reply in Chinese; if English, reply in English. "
                "If unsure, reply in the language most used in conversation."
            )
            vision_content = [
                {"type": "image_url", "image_url": "data:image/jpeg;base64," + encoded}
            ]
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": msg},
                    {"role": "user", "content": vision_content}
                ],
                max_tokens=256,
                temperature=0.2
            )
            desc = response.choices[0].message.content
            return desc if desc else f"[{file_type.capitalize()} received, no description]"
        elif file_type == "audio":
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=file_bytes,
                response_format="text"
            )
            return response if response else "[Audio received, but could not transcribe]"
        elif file_type in ("document", "file"):
            return "[File/document received. Please upload as image/text for better understanding.]"
        else:
            return "[Unrecognized file type]"
    except Exception as e:
        logger.error(f"[AI INTERPRETER] Error: {e}")
        return f"[{file_type.capitalize()} received, but could not interpret]"

# --- Media/Text Extraction, 100% AI-driven ---
def extract_text_from_message(msg):
    text = ""
    raw_media_url = None
    if "body" in msg and msg["body"]:
        text = msg["body"]
    elif msg.get("type") in ("image", "sticker", "video", "audio", "document", "file"):
        raw_media_url = msg.get("mediaUrl") or msg.get("fileUrl")
        text = ai_interpret_file(raw_media_url, msg["type"])
    else:
        text = "[Unrecognized message type]"
    return text, raw_media_url

def send_wassenger_reply(phone, content, device_id, delay_seconds=0, msg_type="text"):
    phone = format_e164(phone)
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"phone": phone, "device": device_id}
    if msg_type == "text":
        payload["message"] = content
    elif msg_type in ("image", "sticker", "video"):
        payload["mediaUrl"] = content
        payload["message"] = ""
        if msg_type != "image":
            payload["type"] = msg_type
    elif msg_type == "audio":
        payload["mediaUrl"] = content
        payload["message"] = ""
        payload["type"] = "audio"
    else:
        payload["message"] = content
    if delay_seconds > 0:
        deliver_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        payload["deliverAt"] = deliver_at.isoformat() + "Z"
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

def get_latest_history(bot_id, customer_phone, session_id, n=50):
    messages = (Message.query
        .filter_by(bot_id=bot_id, customer_phone=customer_phone, session_id=session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
        .all())
    messages = messages[::-1]
    logger.info(f"[DB] History ({len(messages)} messages) loaded.")
    return messages

def send_template_by_id(phone, template_id, device_id):
    tmpl = Template.query.filter_by(template_id=template_id).first()
    if not tmpl:
        logger.warning(f"[TEMPLATE] Template {template_id} not found in DB.")
        return
    parts = json.loads(tmpl.content)
    for obj in parts:
        if obj["type"] == "text":
            send_wassenger_reply(phone, obj["content"], device_id, msg_type="text")
        elif obj["type"] in ["image", "video", "file"]:
            send_wassenger_reply(phone, obj["content"], device_id, msg_type=obj["type"])
        time.sleep(1)

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
    sys_prompt = (bot.system_prompt or "")
    # INSTRUCT: Always reply in the main language of the latest customer message
    sys_prompt += (
        "\nAlways reply in the same language as the customer's latest message. "
        "If unsure, reply in the language most used in the conversation history."
        "\nKeep replies concise, human-like, and split replies using \\n\\n for new message parts."
    )
    prompt = (sys_prompt + "\n" + (tool.prompt or "")) if tool else sys_prompt
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
    for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            reply_accum += chunk.choices[0].delta.content
    logger.info(f"\n[AI REPLY STREAMED]: {reply_accum}")
    return reply_accum

def send_split_messages_wassenger(phone, ai_reply, device_id, bot_id=None, user=None, session_id=None):
    try:
        parsed = json.loads(ai_reply)
        if isinstance(parsed, list):
            for idx, obj in enumerate(parsed):
                msg_type = obj.get("type", "text")
                content = obj.get("content", "")
                send_wassenger_reply(phone, content, device_id, msg_type=msg_type)
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", str(obj))
                if idx < len(parsed) - 1:
                    time.sleep(2)
            return
        if isinstance(parsed, dict) and "template" in parsed:
            send_template_by_id(phone, parsed["template"], device_id)
            return
    except Exception as e:
        logger.error(f"[SEND SPLIT MSGS] Failed to parse AI reply as JSON: {e}")

    parts = [p.strip() for p in ai_reply.split('\n\n') if p.strip()]
    if not parts:
        max_len = 1024
        parts = [ai_reply[i:i+max_len] for i in range(0, len(ai_reply), max_len)]
    for idx, part in enumerate(parts):
        send_wassenger_reply(phone, part, device_id)
        if bot_id and user and session_id:
            save_message(bot_id, user, session_id, "out", part)
        if idx < len(parts) - 1:
            time.sleep(2)

def is_goal_achieved(tool_id, bot_config):
    goal_tools = (bot_config or {}).get("goal_tools", [])
    return tool_id in goal_tools

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
    customer = get_or_create_customer(user_phone)

    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404

    if not msg_text or msg_text.startswith("[Unrecognized"):
        logger.error("[ERROR] Failed to extract text from message.")
        notify_sales_group(bot, f"Failed to process customer message: {raw_media_url}", error=True)
        return jsonify({"error": "Failed to process customer message"}), 500

    save_message(bot.id, user_phone, session_id, "in", msg_text, raw_media_url=raw_media_url)
    history = get_latest_history(bot.id, user_phone, session_id, n=50)
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
    if tool_id and is_goal_achieved(tool_id, bot.config):
        notify_sales_group(bot, f"Goal achieved for customer {user_phone}: {ai_reply}")

    return jsonify({"status": "ok", "ai_reply": ai_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
