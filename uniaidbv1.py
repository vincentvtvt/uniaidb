import os
import re
import time
import json
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base  # updated for SQLAlchemy 2.0+
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime, timedelta
import requests
import anthropic
import tiktoken

# ---- ENV/CONFIG ----
DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
CLAUDE_MAX_TOKENS = 4096

app = Flask(__name__)
Base = declarative_base()

# ---- MODELS ----
class Bot(Base):
    __tablename__ = 'bots'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone_number = Column(String(30), unique=True)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Agent(Base):
    __tablename__ = 'agents'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    agent_type = Column(String(30))
    config = Column(JSON)
    active = Column(Boolean, default=True)

class Workflow(Base):
    __tablename__ = 'workflows'
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey('bots.id'))
    agent_id = Column(Integer, ForeignKey('agents.id'))
    name = Column(String(100))
    flow_config = Column(JSON)
    active = Column(Boolean, default=True)

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey('customers.id'))
    bot_id = Column(Integer, ForeignKey('bots.id'))
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
    session_id = Column(Integer, ForeignKey('sessions.id'))
    sender_type = Column(String(20))
    sender_id = Column(Integer)
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    payload = Column(JSON)

class Tool(Base):
    __tablename__ = 'tools'
    id = Column(Integer, primary_key=True)
    tool_id = Column(String(50), unique=True)
    name = Column(String(100))
    description = Column(Text)
    prompt = Column(Text)
    action_type = Column(String(30))
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class BotTools(Base):
    __tablename__ = 'bot_tools'
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer)
    tool_id = Column(String(50))
    active = Column(Boolean, default=True)

class Template(Base):
    __tablename__ = 'templates'
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer)
    template_id = Column(String(50))
    description = Column(String(255))
    content = Column(JSON, nullable=False)  # List of dicts: [{"type": "text", ...}, ...]
    language = Column(String(10), default='ms')
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ---- DB SETUP ----
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# ---- UTILS ----

def count_tokens(text, model="claude-3-haiku-20240307"):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    return len(tokens)

def get_last_n_messages(session_id, n=20):
    msgs = db_session.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.created_at.desc()).limit(n).all()
    return list(reversed(msgs))

def send_whatsapp_reply(to, text, device_id):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": os.getenv("WASSENGER_API_KEY")}
    payload = {"phone": to, "message": text, "device": device_id}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"Wassenger send error: {e}")

def send_whatsapp_image(to, image_url, device_id):
    url = "https://api.wassenger.com/v1/messages/image"
    headers = {"Content-Type": "application/json", "Token": os.getenv("WASSENGER_API_KEY")}
    payload = {"phone": to, "image": image_url, "device": device_id}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"Wassenger send error (image): {e}")

def send_structured_template(to, template_content, device_id):
    for item in template_content:
        if item["type"] == "text":
            send_whatsapp_reply(to, item["content"], device_id)
        elif item["type"] == "image":
            send_whatsapp_image(to, item["content"], device_id)
        # Add more: "video", "file", etc as needed
        time.sleep(1)  # Avoid spamming user

def get_tools_table_prompt(bot_id):
    tool_links = db_session.query(BotTools).filter(
        BotTools.bot_id == bot_id, BotTools.active == True
    ).all()
    tool_ids = [t.tool_id for t in tool_links]
    tools = db_session.query(Tool).filter(
        Tool.tool_id.in_(tool_ids), Tool.active == True
    ).all()
    table = ""
    for t in tools:
        table += f"{t.tool_id}|{t.name}|{t.description}\n"
    return table.strip()

def get_active_bot(phone_number):
    pn = phone_number.lstrip('+')
    return db_session.query(Bot).filter(Bot.phone_number.like(f"%{pn}")).first()

def get_customer_by_phone(phone_number):
    return db_session.query(Customer).filter(Customer.phone_number == phone_number).first()

def get_latest_session(customer_id, bot_id):
    return db_session.query(Session).filter(Session.customer_id == customer_id, Session.bot_id == bot_id).order_by(Session.started_at.desc()).first()

def save_message(session_id, sender_type, message, meta=None):
    msg = Message(
        session_id=session_id,
        sender_type=sender_type,
        message=message,
        meta=meta or {}
    )
    db_session.add(msg)
    db_session.commit()

def get_template_content(bot_id, template_id):
    tpl = db_session.query(Template).filter(
        Template.bot_id == bot_id,
        Template.template_id == template_id,
        Template.active == True
    ).first()
    return tpl.content if tpl else None

def call_claude(system_prompt, user_message):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text if message and message.content else ""

def build_system_prompt(config, bot_id, context_history=None):
    # Insert dynamic macros into your prompt (e.g., TOOLS_TABLE)
    system_prompt = config.get("system_prompt", "")
    # Strongly enforce JSON output at the end of the prompt:
    json_instruction = """
IMPORTANT: Only reply with valid JSON in the following format. Do not include markdown or commentary, do not include explanations, only output JSON:
{ "template": "template_id", "message": ["text1", "text2"] }
or
{ "message": ["text1", "text2"] }
If no template is used, omit the template field entirely. Only use the message field. 
"""
    if "{{TOOLS_TABLE}}" in system_prompt:
        tools_table = get_tools_table_prompt(bot_id)
        system_prompt = system_prompt.replace("{{TOOLS_TABLE}}", tools_table)
    if context_history:
        system_prompt = f"[Conversation History:]\n{context_history}\n\n{system_prompt}\n\n{json_instruction}"
    else:
        system_prompt = f"{system_prompt}\n\n{json_instruction}"
    return system_prompt

# ---- MAIN CHAT HANDLER ----

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    event_type = data.get('event')
    message_data = data.get('data', {})

    if event_type != 'message:in:new' or message_data.get('meta', {}).get('isGroup'):
        return jsonify({'status': 'ignored'})

    bot_phone = (message_data.get('toNumber') or message_data.get('to') or "").replace('+', '').replace('@c.us', '')
    bot = get_active_bot(bot_phone)
    if not bot:
        return jsonify({'status': 'no_bot_found'})

    config = bot.config or {}
    device_id = config.get("device_id")
    if not device_id:
        return jsonify({'status': 'no_device_id'})

    from_number = (message_data.get('fromNumber') or message_data.get('from', '').lstrip('+')).replace('@c.us','')
    customer = get_customer_by_phone(from_number)
    if not customer:
        customer = Customer(phone_number=from_number, name=from_number, language="und")
        db_session.add(customer)
        db_session.commit()

    session = get_latest_session(customer.id, bot.id)
    if not session:
        session = Session(customer_id=customer.id, bot_id=bot.id, goal="", context={})
        db_session.add(session)
        db_session.commit()

    msg_text = message_data.get('body', '').strip()
    save_message(session.id, 'user', msg_text, {"from": from_number})

    # ---- Fetch last 20 messages for context ----
    messages_for_context = get_last_n_messages(session.id, n=20)
    context_history = ""
    for m in messages_for_context:
        sender = "User" if m.sender_type == "user" else "Assistant"
        context_history += f"{sender}: {m.message}\n"

    # ---- Build prompt with context ----
    manager_prompt = build_system_prompt(config, bot.id, context_history=context_history)

    # ---- Count tokens and log if over ----
    user_tokens = count_tokens(msg_text)
    prompt_tokens = count_tokens(manager_prompt)
    total_tokens = prompt_tokens + user_tokens

    if total_tokens > CLAUDE_MAX_TOKENS:
        print(f"[WARNING] Token limit exceeded! Total: {total_tokens} (Prompt: {prompt_tokens}, User: {user_tokens})")

    # ---- Claude call ----
    ai_result_raw = call_claude(manager_prompt, msg_text)
    print(f"[Claude Raw Output] {ai_result_raw}")  # LOG RAW CLAUDE OUTPUT

    try:
        ai_result = json.loads(ai_result_raw)
    except Exception as e:
        print(f"[Claude JSON error] {e} | Output: {ai_result_raw}")  # LOG JSON ERROR
        ai_result = {}

    # ---- Template (multi-part) reply ----
    if "template" in ai_result:
        template_id = ai_result["template"]
        tpl_content = get_template_content(bot.id, template_id)
        if tpl_content:
            send_structured_template(customer.phone_number, tpl_content, device_id)
            save_message(session.id, 'ai', f"[TEMPLATE:{template_id}]", {})
        else:
            send_whatsapp_reply(customer.phone_number, "Sorry, template not found.", device_id)
        return jsonify({'status': 'template_sent', 'template': template_id})

    # ---- Normal AI message (split if multi-part) ----
    messages = ai_result.get("message") or []
    if isinstance(messages, str):
        messages = [messages]
    if messages:
        for m in messages[:3]:  # Limit to max 3 parts
            send_whatsapp_reply(customer.phone_number, m, device_id)
            time.sleep(1)
        save_message(session.id, 'ai', "\n".join(messages), {})
        return jsonify({'status': 'message_sent'})

    # ---- Fallback ----
    send_whatsapp_reply(customer.phone_number, "ok got it, will get back soon", device_id)
    save_message(session.id, 'ai', "[FALLBACK]", {})
    return jsonify({'status': 'fallback'})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
