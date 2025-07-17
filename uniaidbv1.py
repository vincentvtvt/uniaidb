import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "postgresql://...")  # Edit!
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- SQLAlchemy Models ----------

class Bot(db.Model):
    __tablename__ = 'bots'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    system_prompt = db.Column(db.Text)
    manager_system_prompt = db.Column(db.Text)

class Tool(db.Model):
    __tablename__ = 'tools'
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100))
    description = db.Column(db.Text)
    prompt = db.Column(db.Text)
    action_type = db.Column(db.String(30))
    active = db.Column(db.Boolean)
    created_at = db.Column(db.DateTime, default=func.now())

class BotTool(db.Model):
    __tablename__ = 'bot_tools'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    tool_id = db.Column(db.String(50), nullable=False)  # String!
    active = db.Column(db.Boolean)

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer)
    sender_type = db.Column(db.String(16))
    sender_id = db.Column(db.String(100), nullable=True)
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=func.now())

# ---------- Context Utilities ----------

def save_message(session_id, sender_type, text, sender_id=None):
    m = Message(session_id=session_id, sender_type=sender_type, message=text, sender_id=sender_id)
    db.session.add(m)
    db.session.commit()

def get_context(session_id, limit=20):
    msgs = Message.query.filter_by(session_id=session_id).order_by(Message.created_at.desc()).limit(limit).all()
    return list(reversed([{"role": m.sender_type, "content": m.message} for m in msgs]))

# ---------- Tool Utilities ----------

def get_tools_for_bot(bot_id):
    bot_tools = BotTool.query.filter_by(bot_id=bot_id, active=True).all()
    tool_ids = [bt.tool_id for bt in bot_tools]
    tools = Tool.query.filter(Tool.tool_id.in_(tool_ids), Tool.active == True).all()
    return {t.tool_id: t for t in tools}

def get_default_prompt(bot_id):
    bot = Bot.query.get(bot_id)
    return bot.system_prompt if bot else ""

def get_manager_system_prompt(bot_id):
    bot = Bot.query.get(bot_id)
    return bot.manager_system_prompt if bot else ""

# ---------- AI Orchestration Logic ----------

def orchestrate_ai(msg_text, session_id, bot_id):
    # Gather context and tool info
    context = get_context(session_id)
    tools_dict = get_tools_for_bot(bot_id)
    manager_prompt = get_manager_system_prompt(bot_id)
    system_prompt = get_default_prompt(bot_id)

    # Call your Claude/OpenAI/LLM here (pseudo code)
    # response = call_claude_or_openai(
    #     prompt=manager_prompt, tools=tools_dict, context=context, message=msg_text
    # )

    # For demo, we just echo:
    response = {
        "tool_id": "defaultvpt",  # or pick from tools_dict.keys()
        "reply": f"(AI) You said: {msg_text}"
    }

    # If tool_id is not found, use system_prompt as fallback
    if response["tool_id"] not in tools_dict:
        response["reply"] = system_prompt or "Sorry, no suitable tool. Using default reply."

    return response

# ---------- Wassenger WhatsApp Integration ----------

def send_whatsapp_reply(phone_number, text, device_id=None):
    # Wassenger API call (pseudo code)
    logging.info(f"Send to WhatsApp [{phone_number}]: {text}")
    # TODO: requests.post('https://api.wassenger.com/v1/messages', ...)

# ---------- Flask Webhook ----------

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        payload = request.json
        logging.info(f"[Wassenger Payload]: {payload}")

        # Extract phone/session/other info from payload as needed
        chat = payload.get("data", {}).get("chat", {})
        phone_number = chat.get("id", "")
        session_id = 2  # TODO: Your session mapping logic
        msg_text = payload.get("data", {}).get("body", "")

        # Get bot_id (you should map phone/device to bot_id)
        bot_id = 1

        # Save incoming message
        save_message(session_id, "user", msg_text)

        # Run AI logic
        result = orchestrate_ai(msg_text, session_id, bot_id)

        # Save AI response
        save_message(session_id, "ai", result["reply"])

        # Send back to WhatsApp
        send_whatsapp_reply(phone_number, result["reply"])

        return jsonify({"status": "ok", "result": result}), 200

    except Exception as e:
        logging.exception("Webhook error")
        return jsonify({"status": "error", "reason": str(e)}), 500

# ---------- Run ----------

if __name__ == "__main__":
    logging.basicConfig(level=logging.I
