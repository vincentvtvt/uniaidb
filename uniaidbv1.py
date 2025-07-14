import os
import re
import time
import requests
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
import anthropic

# ---- CONFIG ----
DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")

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

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    customer_id = Column(Integer, ForeignKey('customers.id'))
    agent_id = Column(Integer)
    type = Column(String(30))
    description = Column(Text)
    due_time = Column(DateTime)
    status = Column(String(20), default='pending')
    meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# ---- DB SETUP ----
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# ---- UTILS ----

def detect_language(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    return 'en'

def send_whatsapp_reply(to, text, device_id):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"phone": to, "message": text, "device": device_id}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"Wassenger send error: {e}")

def send_reply_with_delay(receiver, text, device_id, max_parts=3):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    for part in paras[:max_parts]:
        send_whatsapp_reply(receiver, part, device_id)
        time.sleep(1)

# ---- MAIN CHAT HANDLER ----

def get_active_bot(phone_number):
    # Accepts formats like "+60108273799" or "60108273799"
    pn = phone_number.lstrip('+')
    return db_session.query(Bot).filter(Bot.phone_number.like(f"%{pn}")).first()

def get_active_workflow(bot_id):
    return db_session.query(Workflow).filter(Workflow.bot_id == bot_id, Workflow.active == True).first()

def get_agent(agent_id):
    return db_session.query(Agent).filter(Agent.id == agent_id, Agent.active == True).first()

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

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    event_type = data.get('event')
    message_data = data.get('data', {})

    # Only handle new inbound messages, not system/group
    if event_type != 'message:in:new' or message_data.get('meta', {}).get('isGroup'):
        return jsonify({'status': 'ignored'})

    # Find the bot by device phone number
    bot_phone = (message_data.get('toNumber') or message_data.get('to') or "").replace('+', '').replace('@c.us', '')
    bot = get_active_bot(bot_phone)
    if not bot:
        return jsonify({'status': 'no_bot_found'})

    # --- Get device_id from DB for sending reply ---
    device_id = bot.config.get("device_id") if bot and bot.config else None
    if not device_id:
        return jsonify({'status': 'no_device_id'})

    # Get workflow and agent
    workflow = get_active_workflow(bot.id)
    if not workflow:
        return jsonify({'status': 'no_workflow_found'})
    agent = get_agent(workflow.agent_id)
    if not agent:
        return jsonify({'status': 'no_agent_found'})

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
        session = Session(customer_id=customer.id, bot_id=bot.id, goal=workflow.flow_config.get('goal','lead_generation'), context={"workflow": workflow.name})
        db_session.add(session)
        db_session.commit()

    msg_text = message_data.get('body', '').strip()
    save_message(session.id, 'user', msg_text, {"from": from_number})

    # Get workflow steps/prompts from DB
    flow_steps = workflow.flow_config.get('steps', [])
    intro_prompt = next((s["prompt"] for s in flow_steps if s["type"] == "intro"), "Hi, how can I help you?")
    service_menu_prompt = next((s["prompt"] for s in flow_steps if "menu" in s.get("prompt", "").lower()), None)

    # If new session or first message, send intro + menu
    messages = db_session.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at).all()
    if len(messages) == 1:
        send_reply_with_delay(customer.phone_number, intro_prompt, device_id)
        # Send service menu if available (or you can add a separate "menu" step)
        service_menu = db_session.query(Message).filter(Message.session_id == session.id, Message.sender_type == 'ai').first()
        if service_menu:
            send_reply_with_delay(customer.phone_number, service_menu.message, device_id)
        return jsonify({'status': 'intro_sent'})

    # Otherwise, here you can add Claude/AI reply logic using workflow config...

    return jsonify({'status': 'ok'})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
