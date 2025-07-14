import os
import re
import time
import requests
import logging
import json
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

# --- ORM MODELS (Simplified) ---
class Bot(Base):
    __tablename__ = 'bots'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone_number = Column(String(30), unique=True)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    phone_number = Column(String(30), unique=True)
    name = Column(String(50))
    language = Column(String(10))
    meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

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

# --- DB SETUP ---
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# --- Claude & OpenAI Clients ---
openai.api_key = OPENAI_API_KEY
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

# --- Helpers ---
def detect_language(text: str) -> str:
    return 'zh' if re.search(r'[\u4e00-\u9fff]', text) else 'en'

def send_whatsapp_reply(to: str, text: str, device_id: str):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    to_clean = str(to).replace('+','').replace('@c.us','')
    payload = {"phone": to_clean, "message": text, "device": device_id}
    app.logger.debug(f"Sending WhatsApp payload: {payload}")
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        app.logger.info(f"WhatsApp sent: to={to_clean} device={device_id} text='{text[:50]}'")
    except Exception as e:
        app.logger.error(f"Wassenger send error: {e} | Payload: {payload}")

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
    msgs = db_session.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at.desc()).limit(9).all()
    msgs = list(reversed(msgs))
    history = []
    for m in msgs:
        role = 'user' if m.sender_type == 'user' else 'assistant'
        history.append({"role": role, "content": m.message})
    history.append({"role": "user", "content": new_user_input})
    return history

def call_claude_tools(messages: list[dict[str, str]], system_prompt: str) -> dict[str, str]:
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        system=system_prompt,
        messages=messages,
        max_tokens=8192
    )
    text = ''.join(getattr(p, 'text', str(p)) for p in response.content).strip()
    app.logger.debug(f"Claude raw reply: {text}")
    try:
        parsed = json.loads(text)
        app.logger.info(f"Claude JSON reply parsed: {parsed}")
    except Exception as e:
        app.logger.error(f"Claude reply not valid JSON! Error: {e} | Raw: {text}")
        parsed = {"error": "Claude did not return valid JSON", "raw": text}
    return parsed

def call_openai_vision(media_url: str) -> str:
    """Analyzes image or video via OpenAI Vision (GPT-4V) and returns a summary."""
    try:
        result = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe and summarize the content of this image or video in 1-2 sentences."},
                    {"type": "image_url", "image_url": {"url": media_url}}
                ]}
            ],
            max_tokens=8192
        )
        interpretation = result.choices[0].message.content
        app.logger.info(f"OpenAI Vision interpretation: {interpretation[:120]}")
        return interpretation
    except Exception as e:
        app.logger.error(f"OpenAI Vision error: {e}")
        return "Sorry, failed to interpret your image or video."

def call_openai_whisper(audio_url: str) -> str:
    """Transcribes audio using OpenAI Whisper API and returns the text."""
    try:
        # Download the audio file
        audio_data = requests.get(audio_url).content
        with open("/tmp/audio_message.ogg", "wb") as f:
            f.write(audio_data)
        with open("/tmp/audio_message.ogg", "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        transcription = transcript["text"]
        app.logger.info(f"OpenAI Whisper transcription: {transcription[:120]}")
        return transcription
    except Exception as e:
        app.logger.error(f"OpenAI Whisper error: {e}")
        return "Sorry, failed to transcribe your audio."


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

        # --- BOT/DEVICE/PROMPT FROM DATABASE ---
        bot_phone = (message_data.get('toNumber') or message_data.get('to') or "").replace('+', '').replace('@c.us', '')
        bot = get_active_bot(bot_phone)
        if not bot or not bot.config:
            app.logger.error("No bot/config found for phone: " + str(bot_phone))
            return jsonify({'status': 'no_bot_found'})

        device_id = bot.config.get("device_id")
        system_prompt = bot.config.get("system_prompt")
        if not device_id or not system_prompt:
            app.logger.error(f"device_id or system_prompt missing from bot.config: {bot.config}")
            return jsonify({'status': 'no_device_or_prompt'})

        # --- CUSTOMER AND SESSION ---
        from_number = (message_data.get('fromNumber') or message_data.get('from', '').lstrip('+')).replace('@c.us','')
        customer = get_customer_by_phone(from_number)
        if not customer:
            customer = Customer(phone_number=from_number, name=from_number, language=detect_language(message_data.get('body','')))
            db_session.add(customer)
            db_session.commit()
            app.logger.info(f"New customer created: {from_number}")

        session = get_latest_session(customer.id, bot.id)
        if not session:
            session = Session(customer_id=customer.id, bot_id=bot.id, goal="lead_generation", context={"bot": bot.name})
            db_session.add(session)
            db_session.commit()
            app.logger.info(f"New session created: {session.id} for customer {customer.id}")

        # --- MESSAGE HANDLING ---
        msg_type = message_data.get('type', 'text')
        msg_text = message_data.get('body', '').strip()
        media_url = message_data.get('media_url')  # Optional, for future media support

        # --- SAVE INCOMING USER MESSAGE ---
        save_message(
            session.id,
            'user',
            msg_text if msg_text else '[media]',
            {"from": from_number, "type": msg_type, "media_url": media_url}
        )

        # --- BUILD CLAUDE CONTEXT & GET TOOL RECOMMENDATION ---
        messages = build_context_messages(session, msg_text if msg_text else '[media]')
        tool_result = call_claude_tools(messages, system_prompt)

        # --- REPLY TO USER ---
        # For now, echo the Claude output (JSON) as WhatsApp message.
        to_send = json.dumps(tool_result, ensure_ascii=False)
        send_whatsapp_reply(customer.phone_number, to_send, device_id)
        save_message(session.id, 'ai', to_send, {"from": "claude", "tool_reply": tool_result})

        return jsonify({'status': 'reply_sent', 'tool': tool_result})

    except Exception as e:
        app.logger.error(f"Exception in webhook: {e}", exc_info=True)
        return jsonify({'status': 'error', 'detail': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    app.logger.debug("Health endpoint called.")
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
