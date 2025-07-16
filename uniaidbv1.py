import os
import re
import time
import json
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime, timedelta
import requests
import anthropic

# ---- ENV/CONFIG ----
DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")

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
    headers = {"Content-Type": "application/json", "Token": os.getenv("WASSENGER_API_KEY")}
    payload = {"phone": to, "message": text, "device": device_id}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"Wassenger send error: {e}")

def send_split_messages(to, full_text, device_id, max_parts=3):
    parts = [p.strip() for p in full_text.split('\n\n') if p.strip()]
    for part in parts[:max_parts]:
        send_whatsapp_reply(to, part, device_id)


def send_reply_with_delay(receiver, text, device_id, max_parts=3):
    for part in split_message(text, max_parts=max_parts):
        send_whatsapp_reply(receiver, part, device_id)
        time.sleep(1)

def get_tools_table_prompt():
    tools = db_session.query(Tool).filter(Tool.active==True).all()
    table = ""
    for t in tools:
        table += f"{t.tool_id}|{t.description}\n"
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

def notify_sales_group(bot_config, customer, session):
    # Compose message and send to group
    group_id = bot_config.get("wassenger_group_id")
    device_id = bot_config.get("device_id")
    if not group_id or not device_id:
        print("No group_id or device_id for sales notify.")
        return
    info = (
        f"🎉 成交通知\n"
        f"客户姓名: {customer.name}\n"
        f"手机号: {customer.phone_number}\n"
        f"产品: {session.goal or '未填写'}\n"
        f"成交时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"请尽快跟进！"
    )
    send_whatsapp_reply(group_id, info, device_id)

def get_next_weekdays():
    today = datetime.now()
    return [
        (today + timedelta(days=i)).strftime("%m月%d日")
        for i in range(1, 8)
    ]

# ---- CLAUDE (ANTHROPIC) ----

def call_claude(system_prompt, user_message):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text if message and message.content else ""

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
        customer = Customer(phone_number=from_number, name=from_number, language=detect_language(message_data.get('body','')))
        db_session.add(customer)
        db_session.commit()

    session = get_latest_session(customer.id, bot.id)
    if not session:
        session = Session(customer_id=customer.id, bot_id=bot.id, goal="", context={})
        db_session.add(session)
        db_session.commit()

    msg_text = message_data.get('body', '').strip()
    save_message(session.id, 'user', msg_text, {"from": from_number})

    # --- Manager AI picks tool
    manager_prompt = config.get("manager_system_prompt", "")
    manager_prompt = manager_prompt.replace("{{TOOLS_TABLE}}", get_tools_table_prompt())
    tool_pick = call_claude(manager_prompt, msg_text)
    try:
        tool_json = json.loads(tool_pick)
        tool_id = tool_json.get("TOOLS", "Default")
    except Exception as e:
        tool_id = "Default"

    # ---- If "saleswon", notify group
    if tool_id.lower() == "saleswon":
        notify_sales_group(config, customer, session)
        send_whatsapp_reply(customer.phone_number, "感谢您的配合，后续团队会尽快跟进并致电您！", device_id)
        session.status = "closed"
        session.ended_at = datetime.now()
        db_session.commit()
        return jsonify({'status': 'saleswon_notified'})

    # ---- Normal agent/AI reply (you can expand this for your specific Claude reply)
    # This is where your "system_prompt" can be used for natural reply (e.g. for default flow)
    agent_prompt = config.get("system_prompt", "")
    ai_reply = call_claude(agent_prompt, msg_text)
    send_reply_with_delay(customer.phone_number, ai_reply, device_id, max_parts=3)
    save_message(session.id, 'ai', ai_reply, {})

    return jsonify({'status': 'ok', 'tool': tool_id})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
