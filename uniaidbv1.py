import os
import re
import time
import logging
import requests
import json
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
import anthropic

# ---- CONFIG ----
DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")
WASSENGER_API_URL = "https://api.wassenger.com/v1/messages"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-bot")

app = Flask(__name__)
Base = declarative_base()

# ---- ORM MODELS ----
class Bot(Base):
    __tablename__ = 'bots'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone_number = Column(String(30), unique=True)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Tool(Base):
    __tablename__ = 'tools'
    id = Column(Integer, primary_key=True)
    tool_id = Column(String(50))
    name = Column(String(50))
    description = Column(Text)
    prompt = Column(Text)
    action_type = Column(String(50))
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class BotTool(Base):
    __tablename__ = 'bot_tools'
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey('bots.id'))
    tool_id = Column(Integer, ForeignKey('tools.id'))
    active = Column(Boolean, default=True)

class Workflow(Base):
    __tablename__ = 'workflows'
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey('bots.id'))
    agent_id = Column(Integer)
    name = Column(String(100))
    flow_config = Column(JSON)
    active = Column(Boolean, default=True)

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer)
    bot_id = Column(Integer)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    status = Column(String(20), default='open')
    goal = Column(String(50))
    context = Column(JSON)

class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    phone_number = Column(String(30), unique=True)
    name = Column(String(50))
    language = Column(String(10))
    meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    sender_type = Column(String(20))
    sender_id = Column(Integer)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    payload = Column(JSON)

# ---- DB SETUP ----
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# ---- UTILS ----

def detect_language(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    return 'en'

def send_whatsapp_reply(to, text, device_id):
    url = WASSENGER_API_URL
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"phone": to, "message": text, "device": device_id}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
        logger.info(f"Sent WhatsApp: {to} — {text[:80]}...")
    except Exception as e:
        logger.error(f"Wassenger send error: {e}")

def split_reply_ai(full_reply, max_parts=3):
    # Ask Claude to split reply for WhatsApp
    if len(full_reply) < 700:
        return [full_reply]
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    system = "You are a WhatsApp reply splitting assistant. Split the following reply into 2 or 3 natural, human-sounding parts/messages (each under 700 characters if possible). Avoid splitting in the middle of a sentence. Return as a JSON array of strings."
    prompt = f"{system}\n\nText:\n{full_reply}"
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        arr = resp.content[0].text if isinstance(resp.content, list) else resp.content
        arr = arr.strip()
        arr = arr[arr.find('['):]  # To handle possible extra text before array
        arr = eval(arr)
        if isinstance(arr, list):
            return arr[:max_parts]
    except Exception as e:
        logger.warning(f"AI split failed, fallback to sentence splitting. Err: {e}")
    # Fallback: naive split by '. '
    sentences = full_reply.split('. ')
    out, current = [], ""
    for s in sentences:
        if len(current) + len(s) > 680 and current:
            out.append(current.strip())
            current = ""
        current += s + ". "
    if current:
        out.append(current.strip())
    return out[:max_parts]

def send_reply_with_delay(receiver, text, device_id, max_parts=3):
    paras = split_reply_ai(text, max_parts)
    for part in paras:
        send_whatsapp_reply(receiver, part, device_id)
        time.sleep(1)

def get_active_bot(phone_number):
    pn = phone_number.lstrip('+')
    return db_session.query(Bot).filter(Bot.phone_number.like(f"%{pn}")).first()

def get_active_tools(bot_id):
    return (
        db_session.query(Tool)
        .join(BotTool, Tool.id == BotTool.tool_id)
        .filter(BotTool.bot_id == bot_id, BotTool.active == True, Tool.active == True)
        .all()
    )

def get_customer_by_phone(phone_number):
    return db_session.query(Customer).filter(Customer.phone_number == phone_number).first()

def get_latest_session(customer_id, bot_id):
    return db_session.query(Session).filter(Session.customer_id == customer_id, Session.bot_id == bot_id, Session.status == 'open').order_by(Session.started_at.desc()).first()

def save_message(session_id, sender_type, message, meta=None, sender_id=None):
    msg = Message(
        session_id=session_id,
        sender_type=sender_type,
        sender_id=sender_id,
        message=message,
        meta=meta or {}
    )
    db_session.add(msg)
    db_session.commit()

def build_manager_prompt(base_prompt, tools):
    tool_rows = [f"{t.tool_id}|{t.description}" for t in tools]
    tool_table = "ID|描述\n" + "\n".join(tool_rows)
    return base_prompt.replace("{{TOOLS_TABLE}}", tool_table)

def call_claude_manager(prompt, latest_msg):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        system=prompt,
        messages=[{"role": "user", "content": latest_msg}]
    )
    raw = resp.content[0].text if isinstance(resp.content, list) else resp.content
    try:
        tool_decision = json.loads(raw)
    except Exception:
        # Try to extract JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        tool_decision = json.loads(match.group(0)) if match else {"TOOLS": "Default"}
    logger.info(f"Manager selected tool: {tool_decision}")
    return tool_decision

def call_claude_reply(system_prompt, chat_history):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=system_prompt,
        messages=chat_history
    )
    return resp.content[0].text if isinstance(resp.content, list) else resp.content

# ---- MAIN CHAT HANDLER ----

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    logger.info(f"Received webhook: {data}")

    event_type = data.get('event')
    message_data = data.get('data', {})

    if event_type != 'message:in:new' or message_data.get('meta', {}).get('isGroup'):
        logger.info("Event not inbound user message or is group; ignored.")
        return jsonify({'status': 'ignored'})

    bot_phone = (message_data.get('toNumber') or message_data.get('to') or "").replace('+', '').replace('@c.us', '')
    bot = get_active_bot(bot_phone)
    if not bot:
        logger.error("No bot found for device number")
        return jsonify({'status': 'no_bot_found'})

    config = bot.config or {}
    device_id = config.get("device_id")
    manager_base_prompt = config.get("manager_system_prompt")
    system_prompt = config.get("system_prompt")

    if not device_id or not manager_base_prompt or not system_prompt:
        logger.error("Device ID or prompts missing in bot config")
        return jsonify({'status': 'bad_bot_config'})

    # Get customer info (or create if not exists)
    from_number = (message_data.get('fromNumber') or message_data.get('from', '').lstrip('+')).replace('@c.us','')
    customer = get_customer_by_phone(from_number)
    if not customer:
        customer = Customer(phone_number=from_number, name=from_number, language=detect_language(message_data.get('body','')))
        db_session.add(customer)
        db_session.commit()

    # Find latest session or create a new one
    session = get_latest_session(customer.id, bot.id)
    if not session:
        session = Session(customer_id=customer.id, bot_id=bot.id, goal="lead_generation", context={})
        db_session.add(session)
        db_session.commit()

    msg_text = message_data.get('body', '').strip()
    save_message(session.id, 'user', msg_text, {"from": from_number}, sender_id=customer.id)

    # --- MANAGER AI: Tool Selection ---
    tools = get_active_tools(bot.id)
    manager_prompt = build_manager_prompt(manager_base_prompt, tools)
    tool_decision = call_claude_manager(manager_prompt, msg_text)
    selected_tool = tool_decision.get("TOOLS", "Default")

    # --- CUSTOMER AI: Generate human reply based on selected tool ---
    # Prepare chat history for Claude
    previous_msgs = db_session.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at).all()
    chat_history = []
    for m in previous_msgs[-6:]:
        role = "user" if m.sender_type == "user" else "assistant"
        chat_history.append({"role": role, "content": m.message})

    # Pick system prompt based on tool
    if selected_tool == "Default":
        final_prompt = system_prompt
    else:
        # Try to find the specific tool's prompt, fallback to system_prompt
        t = next((x for x in tools if x.tool_id == selected_tool), None)
        final_prompt = (t.prompt if t and t.prompt else system_prompt)

    try:
        ai_reply = call_claude_reply(final_prompt, chat_history)
        logger.info(f"Claude AI reply: {ai_reply}")

        send_reply_with_delay(customer.phone_number, ai_reply, device_id)
        save_message(session.id, 'ai', ai_reply, {"to": customer.phone_number}, sender_id=bot.id)

        return jsonify({'status': 'ok', 'tool': selected_tool, 'ai_reply': ai_reply[:80]})
    except Exception as e:
        logger.error(f"Claude AI call/send error: {e}")
        send_whatsapp_reply(customer.phone_number, "Sorry, the assistant is currently unavailable.", device_id)
        return jsonify({'status': 'ai_error', 'detail': str(e)})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
