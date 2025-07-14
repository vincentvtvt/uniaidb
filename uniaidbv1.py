import os
import re
import time
import requests
import logging
from typing import Optional, List, Dict, Any
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime

import anthropic
import openai

# --- CONFIGURATION ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Base = declarative_base()

# --- ORM MODELS ---
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

# --- DB SETUP ---
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# --- OpenAI & Claude Clients ---
openai.api_key = OPENAI_API_KEY
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# --- Helper Functions ---
def detect_language(text: str) -> str:
    return 'zh' if re.search(r'[\u4e00-\u9fff]', text) else 'en'

def send_whatsapp_reply(to: str, text: str, device_id: str):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"phone": to, "message": text, "device": device_id}
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        app.logger.info(f"WhatsApp sent: to={to} device={device_id} text='{text[:50]}'")
    except Exception as e:
        app.logger.error(f"Wassenger send error: {e}")

def send_reply_with_delay(receiver: str, text: str, device_id: str, max_parts: int = 3):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    for part in paras[:max_parts]:
        send_whatsapp_reply(receiver, part, device_id)
        time.sleep(1)

def get_active_bot(phone_number: str) -> Optional[Bot]:
    pn = phone_number.lstrip('+')
    bot = db_session.query(Bot).filter(Bot.phone_number.like(f"%{pn}")).first()
    app.logger.debug(f"get_active_bot({phone_number}) -> {bot}")
    return bot

def get_active_workflow(bot_id: int) -> Optional[Workflow]:
    workflow = db_session.query(Workflow).filter(Workflow.bot_id == bot_id, Workflow.active == True).first()
    app.logger.debug(f"get_active_workflow({bot_id}) -> {workflow}")
    return workflow

def get_agent(agent_id: int) -> Optional[Agent]:
    agent = db_session.query(Agent).filter(Agent.id == agent_id, Agent.active == True).first()
    app.logger.debug(f"get_agent({agent_id}) -> {agent}")
    return agent

def get_customer_by_phone(phone_number: str) -> Optional[Customer]:
    cust = db_session.query(Customer).filter(Customer.phone_number == phone_number).first()
    app.logger.debug(f"get_customer_by_phone({phone_number}) -> {cust}")
    return cust

def get_latest_session(customer_id: int, bot_id: int) -> Optional[Session]:
    session = db_session.query(Session).filter(Session.customer_id == customer_id, Session.bot_id == bot_id).order_by(Session.started_at.desc()).first()
    app.logger.debug(f"get_latest_session({customer_id}, {bot_id}) -> {session}")
    return session

def save_message(session_id: int, sender_type: str, message: str, meta: Optional[Dict[str, Any]] = None):
    try:
        msg = Message(
            session_id=session_id,
            sender_type=sender_type,
            message=message,
            meta=meta or {}
        )
        db_session.add(msg)
        db_session.commit()
        app.logger.info(f"Message saved: {sender_type} session_id={session_id}")
    except Exception as e:
        app.logger.error(f"Error saving message: {e}")

def build_context_messages(session: Session, new_user_input: str) -> List[Dict[str, str]]:
    # Build the last 10 messages for context
    msgs = db_session.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at.desc()).limit(9).all()
    msgs = list(reversed(msgs))
    history = []
    for m in msgs:
        role = 'user' if m.sender_type == 'user' else 'assistant'
        history.append({"role": role, "content": m.message})
    history.append({"role": "user", "content": new_user_input})
    return history

def call_claude_api(messages: List[Dict[str, str]]) -> str:
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        system="You are an expert sales and service assistant for Ventopia. Respond in the user's language, be polite, and follow the workflow.",
        messages=messages,
        max_tokens=1024
    )
    reply = ''.join(getattr(p, 'text', str(p)) for p in response.content).strip()
    app.logger.debug(f"Claude reply: {reply[:120]}")
    return reply

def call_openai_vision(media_url: str) -> str:
    try:
        # Download the image/video
        file_bytes = requests.get(media_url).content
        app.logger.info("Media file downloaded for OpenAI Vision.")
        # Call OpenAI Vision
        result = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe the content of this image or video, and summarize it in one or two sentences."},
                    {"type": "image_url", "image_url": {"url": media_url}}
                ]}
            ],
            max_tokens=512
        )
        interpretation = result.choices[0].message.content
        app.logger.info(f"OpenAI Vision interpretation: {interpretation[:120]}")
        return interpretation
    except Exception as e:
        app.logger.error(f"OpenAI Vision error: {e}")
        return "Sorry, failed to interpret your media."

# --- WEBHOOK HANDLER ---
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json(force=True)
        app.logger.debug(f"Incoming payload: {data}")

        event_type = data.get('event')
        message_data = data.get('data', {})

        # Only handle new inbound messages, not group/system
        if event_type != 'message:in:new' or message_data.get('meta', {}).get('isGroup'):
            app.logger.info("Ignored event or group message.")
            return jsonify({'status': 'ignored'})

        # Get bot and device_id
        bot_phone = (message_data.get('toNumber') or message_data.get('to') or "").replace('+', '').replace('@c.us', '')
        bot = get_active_bot(bot_phone)
        if not bot:
            app.logger.error("No bot found for phone: " + str(bot_phone))
            return jsonify({'status': 'no_bot_found'})

        device_id = bot.config.get("device_id") if bot and bot.config else None
        app.logger.debug(f"Device ID loaded: {device_id}")
        if not device_id:
            app.logger.error("No device ID found in bot.config.")
            return jsonify({'status': 'no_device_id'})

        # Get workflow, agent, customer, session
        workflow = get_active_workflow(bot.id)
        if not workflow:
            app.logger.error("No workflow found for bot_id=" + str(bot.id))
            return jsonify({'status': 'no_workflow_found'})
        agent = get_agent(workflow.agent_id)
        if not agent:
            app.logger.error("No agent found for agent_id=" + str(workflow.agent_id))
            return jsonify({'status': 'no_agent_found'})

        from_number = (message_data.get('fromNumber') or message_data.get('from', '').lstrip('+')).replace('@c.us','')
        customer = get_customer_by_phone(from_number)
        if not customer:
            customer = Customer(phone_number=from_number, name=from_number, language=detect_language(message_data.get('body','')))
            db_session.add(customer)
            db_session.commit()
            app.logger.info(f"New customer created: {from_number}")

        session = get_latest_session(customer.id, bot.id)
        if not session:
            session = Session(customer_id=customer.id, bot_id=bot.id, goal=workflow.flow_config.get('goal','lead_generation'), context={"workflow": workflow.name})
            db_session.add(session)
            db_session.commit()
            app.logger.info(f"New session created: {session.id} for customer {customer.id}")

        # Message handling: text, image, video
        msg_type = message_data.get('type', 'text')  # Check your webhook for real field name!
        msg_text = message_data.get('body', '').strip()
        media_url = message_data.get('media_url')  # Confirm the actual media field from your payload!

        # Save every incoming user message
        save_message(
                session.id,
                'user',
                msg_text if msg_text else '[media]',
                {"from": from_number, "type": msg_type, "media_url": media_url}
        )
