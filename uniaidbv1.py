import os
import re
import time
import logging
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
            max_tokens=1024,
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

def get_active_workflow(bot_id):
    return db_session.query(Workflow).filter(Workflow.bot_id == bot_id, Workflow.active == True).first()

def get_agent(agent_id):
    return db_session.query(Agent).filter(Agent.id == agent_id, Agent.active == True).first()

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

    device_id = bot.config.get("device_id") if bot and bot.config else None
    if not device_id:
        logger.error("Device ID missing in bot config")
        return jsonify({'status': 'no_device_id'})

    workflow = get_active_workflow(bot.id)
    if not workflow:
        logger.error("No workflow found for bot")
        return jsonify({'status': 'no_workflow_found'})
    agent = get_agent(workflow.agent_id)
    if not agent:
        logger.error("No agent found for workflow")
        return jsonify({'status': 'no_agent_found'})

    from_number = (message_data.get('fromNumber') or message_data.get('from', '').lstrip('+')).replace('@c.us','')
    customer = get_customer_by_phone(from_number)
    if not customer:
        customer = Customer(phone_number=from_number, name=from_number, language=detect_language(message_data.get('body','')))
        db_session.add(customer)
        db_session.commit()

    session = get_latest_session(customer.id, bot.id)
    if not session:
        session = Session(customer_id=customer.id, bot_id=bot.id, goal=workflow.flow_config.get('goal','lead_generation'), context={"workflow": workflow.name})
        db_session.add(session)
        db_session.commit()

    msg_text = message_data.get('body', '').strip()
    save_message(session.id, 'user', msg_text, {"from": from_number}, sender_id=customer.id)

    # --- Workflow steps: intro if first message only
    messages = db_session.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at).all()
    if len(messages) == 1:
        flow_steps = workflow.flow_config.get('steps', [])
        intro_prompt = next((s["prompt"] for s in flow_steps if s["type"] == "intro"), "Hi, how can I help you?")
        logger.info(f"Sending intro: {intro_prompt}")
        send_reply_with_delay(customer.phone_number, intro_prompt, device_id)
        save_message(session.id, 'ai', intro_prompt, {"to": customer.phone_number}, sender_id=bot.id)
        return jsonify({'status': 'intro_sent'})

    # ---- AI reply logic for further messages ----
    try:
        # Collect last N messages for context (for better Claude results)
        previous = db_session.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at).all()
        history = []
        for m in previous[-6:]:
            role = "user" if m.sender_type == "user" else "assistant"
            history.append({"role": role, "content": m.message})

        system_prompt = (bot.config.get("system_prompt") if bot and bot.config else "You are a helpful, human-like WhatsApp assistant. Respond naturally, split into 2–3 messages if reply is long.")
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            messages=[{"role": "system", "content": system_prompt}] + history
        )
        ai_reply = response.content[0].text if isinstance(response.content, list) else response.content
        logger.info(f"Claude AI reply: {ai_reply}")

        send_reply_with_delay(customer.phone_number, ai_reply, device_id)
        save_message(session.id, 'ai', ai_reply, {"to": customer.phone_number}, sender_id=bot.id)

        return jsonify({'status': 'ok', 'ai_reply': ai_reply[:80]})
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
