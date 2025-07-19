import os, re, json, time, logging, requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import anthropic, openai  # Use OpenAI for vision/audio, Claude for text

# ========== ENV & LOGGING ==========
DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAI")

# ========== FLASK & DB SETUP ==========
app = Flask(__name__)
Base = declarative_base()
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# ========== ORM MODELS ==========
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
    sender_type = Column(String(20))  # user/ai
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    payload = Column(JSON)

class Tool(Base):
    __tablename__ = 'tools'
    id = Column(Integer, primary_key=True)
    tool_id = Column(String(30), unique=True)
    name = Column(String(50))
    description = Column(Text)
    prompt = Column(Text)  # Can be JSON or plain text
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ========== UTILS ==========
def get_active_bot(phone_number):
    pn = phone_number.lstrip('+')
    bot = db_session.query(Bot).filter(Bot.phone_number.like(f"%{pn}")).first()
    logger.info(f"[DB] Bot lookup by phone: {phone_number} => {bot}")
    return bot

def get_customer_by_phone(phone_number):
    customer = db_session.query(Customer).filter(Customer.phone_number == phone_number).first()
    return customer

def get_or_create_customer(phone_number, name=None, language=None):
    customer = get_customer_by_phone(phone_number)
    if not customer:
        customer = Customer(phone_number=phone_number, name=name or phone_number, language=language)
        db_session.add(customer)
        db_session.commit()
    return customer

def get_latest_session(customer_id, bot_id):
    return db_session.query(Session).filter(Session.customer_id == customer_id, Session.bot_id == bot_id, Session.status == 'open').order_by(Session.started_at.desc()).first()

def start_new_session(customer_id, bot_id, goal=None):
    session = Session(customer_id=customer_id, bot_id=bot_id, goal=goal, status='open', started_at=datetime.now())
    db_session.add(session)
    db_session.commit()
    return session

def save_message(session_id, sender_type, message, meta=None, payload=None):
    msg = Message(session_id=session_id, sender_type=sender_type, message=message, meta=meta or {}, payload=payload or {})
    db_session.add(msg)
    db_session.commit()
    logger.info(f"[DB] Saved message ({sender_type}) for session {session_id}: {message}")

def load_history(customer_id, bot_id, limit=50):
    sessions = db_session.query(Session).filter(Session.customer_id == customer_id, Session.bot_id == bot_id).order_by(Session.started_at.desc()).all()
    if not sessions: return []
    session_ids = [s.id for s in sessions]
    msgs = db_session.query(Message).filter(Message.session_id.in_(session_ids)).order_by(Message.created_at.desc()).limit(limit).all()
    return list(reversed([{"sender": m.sender_type, "text": m.message, "meta": m.meta, "created_at": m.created_at.isoformat()} for m in msgs]))

def close_session(session):
    session.status = 'closed'
    session.ended_at = datetime.now()
    db_session.commit()
    logger.info(f"[SESSION] Closed session {session.id}")

def get_tools_for_bot(bot_id):
    return db_session.query(Tool).filter(Tool.active==True).all()

def get_tool_by_id(tool_id):
    return db_session.query(Tool).filter(Tool.tool_id == tool_id).first()

def send_wassenger_reply(phone, text, device_id):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"phone": phone if phone.startswith("+") else f"+{phone}", "message": text, "device": device_id}
    r = requests.post(url, json=payload, headers=headers)
    logger.info(f"[WASSENGER] Sent: {text} to {phone}, status: {r.status_code}")

def send_wassenger_template(phone, template_json, device_id):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    # e.g. template_json = [{"type": "text", "content": "..."}, {"type": "image", ...}]
    payload = {"phone": phone if phone.startswith("+") else f"+{phone}", "messages": template_json, "device": device_id}
    r = requests.post(url, json=payload, headers=headers)
    logger.info(f"[WASSENGER] Sent template: {template_json} to {phone}, status: {r.status_code}")

def split_and_send_reply(phone, reply, device_id, bot_id=None, user=None, session_id=None):
    # Convert JSON-like reply to plain text
    parts = []
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except Exception:
            pass
    if isinstance(reply, dict) and "message" in reply:
        message_list = reply["message"]
        if isinstance(message_list, list):
            flat = "\n\n".join([str(m).strip() for m in message_list if m.strip()])
            parts = [p.strip() for p in flat.split('\n\n') if p.strip()]
        else:
            parts = [str(message_list)]
    elif isinstance(reply, list):
        parts = [str(m).strip() for m in reply if str(m).strip()]
    else:
        reply_str = reply if isinstance(reply, str) else str(reply)
        parts = [p.strip() for p in reply_str.split('\n\n') if p.strip()]
    for idx, part in enumerate(parts[:3]):
        send_wassenger_reply(phone, part, device_id)
        if bot_id and user and session_id:
            save_message(session_id, "ai", part)
        if idx < len(parts[:3]) - 1:
            time.sleep(1)

def download_and_interpret_media(media_url, file_type="image"):
    headers = {"Token": WASSENGER_API_KEY}
    r = requests.get(media_url, headers=headers)
    file_bytes = r.content
    if file_type == "image":
        openai.api_key = OPENAI_API_KEY
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a sticker/meme interpreter, summarize what this image expresses."},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": media_url}}]}
            ],
            max_tokens=512,
        )
        ai_reply = response.choices[0].message.content.strip()
        return ai_reply
    # Audio/other can use whisper
    return "[Media file received]"

def ai_language_detect(history):
    # Use Claude/OpenAI to infer language from context, prefer last user msg
    latest_user_msg = None
    for h in reversed(history):
        if h["sender"] == "user":
            latest_user_msg = h["text"]
            break
    if not latest_user_msg:
        return "en"
    # Use OpenAI for quick detection
    openai.api_key = OPENAI_API_KEY
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "What is the language of the following message? Only answer as ISO code (en, zh, ms, etc)."}, {"role": "user", "content": latest_user_msg}],
        max_tokens=1,
    )
    lang = completion.choices[0].message.content.strip().lower()
    if lang not in ["en", "zh", "ms"]:
        lang = "en"
    return lang

def call_claude(system_prompt, history, user_message, language=None):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    # Compose context
    history_str = ""
    for msg in history[-50:]:
        if msg['sender'] == 'user':
            history_str += f"User: {msg['text']}\n"
        else:
            history_str += f"Bot: {msg['text']}\n"
    final_prompt = f"{system_prompt}\nCurrent conversation (last 50 messages):\n{history_str}\nUser message: {user_message}"
    if language:
        final_prompt += f"\nCustomer language: {language}\n"
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": final_prompt}]
    )
    return message.content[0].text if message and message.content else ""

# ========== FLASK WEBHOOK ==========
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    msg = data.get('data', {})
    logger.info(f"[INCOMING] WhatsApp message: {json.dumps(msg)[:500]}")
    event_type = data.get('event')

    # Ignore groups/system
    if event_type != 'message:in:new' or msg.get('meta', {}).get('isGroup'):
        return jsonify({'status': 'ignored'})

    bot_phone = (msg.get('toNumber') or msg.get('to', '')).replace('+', '').replace('@c.us', '')
    bot = get_active_bot(bot_phone)
    if not bot or not bot.config or not bot.config.get("device_id"):
        return jsonify({"error": "No bot/device"})
    device_id = bot.config["device_id"]

    from_number = (msg.get('fromNumber') or msg.get('from', '')).replace('@c.us', '').replace('+','')
    user_name = msg.get("chat", {}).get("contact", {}).get("name", from_number)
    customer = get_or_create_customer(from_number, user_name)
    session = get_latest_session(customer.id, bot.id)
    if not session:
        session = start_new_session(customer.id, bot.id)
    user_message = msg.get("body", "") or ""
    media_msg = ""
    # If it's media, download and run vision/whisper
    if msg.get("type") in ["image", "sticker", "video", "audio", "document"]:
        links = msg.get("media", {}).get("links", {})
        download_url = None
        if "download" in links:
            download_url = "https://api.wassenger.com" + links["download"]
        elif "resource" in links:
            download_url = "https://api.wassenger.com" + links["resource"]
        if download_url:
            try:
                media_msg = download_and_interpret_media(download_url, file_type=msg.get("type", "image"))
                user_message = f"[{msg.get('type').capitalize()}] {media_msg}"
            except Exception as e:
                logger.error(f"[AI INTERPRETER] Error: {e}")
                user_message = f"[{msg.get('type').capitalize()} received, but could not interpret]"
    save_message(session.id, "user", user_message, meta={"raw": msg})

    # Load last 50 history (plain, AI will decide context)
    history = load_history(customer.id, bot.id, 50)
    language = ai_language_detect(history)

    # Manager AI: choose tool/goal, pass tools list/goals, system prompt from DB/config
    manager_system_prompt = bot.config.get("manager_system_prompt", "You are the manager.")
    tools = get_tools_for_bot(bot.id)
    tools_table = "\n".join([f"{t.tool_id}|{t.description}" for t in tools])
    goal_tools = "\n".join([t.tool_id for t in tools if "88" in t.tool_id])
    manager_prompt_final = f"{manager_system_prompt}\n<TOOLS>\n{tools_table}\n</TOOLS>\n<GoalTools>\n{goal_tools}\n</GoalTools>\n"
    ai_decision = call_claude(manager_prompt_final, history, user_message, language=language)
    logger.info(f"[AI DECISION] manager_system_prompt: {manager_prompt_final[:300]}...")
    logger.info(f"[AI DECISION] Tool chosen: {ai_decision}")

    # Parse tool output: {"TOOLS": "something"}
    try:
        tool_result = json.loads(ai_decision)
        tool_id = tool_result.get("TOOLS", "Default")
    except Exception:
        tool_id = "Default"

    # Tool action: if template/media/attachments, use tool.prompt as JSON; else normal
    tool = get_tool_by_id(tool_id)
    if tool and tool.prompt:
        try:
            prompt_obj = json.loads(tool.prompt)
            # E.g. {"template": "bf_product", "message": [...], "mediaUrl": "..."}
            if "template" in prompt_obj or "mediaUrl" in prompt_obj:
                send_wassenger_template(customer.phone_number, prompt_obj, device_id)
                save_message(session.id, "ai", json.dumps(prompt_obj))
            else:
                split_and_send_reply(customer.phone_number, prompt_obj.get("message", ""), device_id, bot.id, customer.id, session.id)
        except Exception:
            # Not JSON: just split and send as text
            split_and_send_reply(customer.phone_number, tool.prompt, device_id, bot.id, customer.id, session.id)
    else:
        # AI direct reply: call Claude as normal chat AI
        system_prompt = bot.config.get("system_prompt", "You are Richelle, answer in max 3 concise sentences, split by paragraph, never use JSON.")
        ai_reply = call_claude(system_prompt, history, user_message, language=language)
        split_and_send_reply(customer.phone_number, ai_reply, device_id, bot.id, customer.id, session.id)

    # If the tool is a goal/close (contains "88"), auto-close session
    if "88" in tool_id:
        close_session(session)

    return jsonify({"status": "ok"})

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
