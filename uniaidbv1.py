import os
import requests
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger("unibot")

# ---- Flask & DB Setup ----
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "postgresql://YOUR_DB_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---- Models ----
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
    tool_id = db.Column(db.Integer)
    active = db.Column(db.Boolean)

# ---- Prompt Logic ----
def get_reply_prompt(bot_id, action_type="default"):
    log.info(f"Fetching bot for bot_id={bot_id}")
    bot = db.session.query(Bot).filter_by(id=bot_id).first()
    log.info(f"Bot fetch result: {bot}")
    if not bot:
        log.error("Bot not found!")
        return "Error: Bot not found."

    log.info(f"Fetching tool for bot_id={bot_id} and action_type={action_type}")
    tool = (
        db.session.query(Tool)
        .join(BotTool, Tool.id == BotTool.tool_id)
        .filter(
            BotTool.bot_id == bot_id,
            Tool.active == True,
            BotTool.active == True,
            Tool.action_type == action_type
        )
        .first()
    )
    log.info(f"Tool fetch result: {tool}")

    if tool and tool.prompt:
        assembled_prompt = f"{bot.system_prompt.strip() if bot.system_prompt else ''}\n\n{tool.prompt.strip()}"
        log.info(f"Prompt assembly (system_prompt + tool.prompt): {assembled_prompt}")
        return assembled_prompt
    else:
        if bot.system_prompt and bot.system_prompt.strip():
            log.info(f"Prompt fallback to system_prompt: {bot.system_prompt.strip()}")
            return bot.system_prompt.strip()
        elif bot.manager_system_prompt and bot.manager_system_prompt.strip():
            log.info(f"Prompt fallback to manager_system_prompt: {bot.manager_system_prompt.strip()}")
            return bot.manager_system_prompt.strip()
        else:
            log.error("No prompt configured at all.")
            return "No prompt configured."

# ---- Wassenger Send Function ----
def send_wassenger_message(phone, text, device_id=None):
    api_url = "https://api.wassenger.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "Token": os.getenv("WASSENGER_API_KEY")
    }
    payload = {
        "phone": phone,
        "message": text
    }
    if device_id:
        payload["device"] = device_id
    log.info(f"Sending to Wassenger. Payload: {payload}")
    try:
        resp = requests.post(api_url, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        log.info(f"Wassenger response: {result}")
        return result
    except Exception as e:
        log.error(f"Wassenger send error: {e}")
        return None

# ---- API Endpoints ----

@app.route("/get_prompt", methods=["GET"])
def get_prompt():
    bot_id = request.args.get("bot_id", type=int)
    action_type = request.args.get("action_type", default="default", type=str)
    log.info(f"API /get_prompt called: bot_id={bot_id}, action_type={action_type}")
    prompt = get_reply_prompt(bot_id, action_type)
    log.info(f"API /get_prompt result: {prompt}")
    return jsonify({"prompt": prompt})

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.json
    bot_id = data.get("bot_id")
    to_phone = data.get("phone")
    action_type = data.get("action_type", "default")
    device_id = data.get("device_id")
    log.info(f"API /send_message called: bot_id={bot_id}, phone={to_phone}, action_type={action_type}, device_id={device_id}")

    prompt = get_reply_prompt(bot_id, action_type)
    log.info(f"Prepared prompt to send: {prompt}")

    result = send_wassenger_message(to_phone, prompt, device_id=device_id)
    log.info(f"Message send result: {result}")

    return jsonify({
        "wassenger_result": result,
        "message_sent": prompt
    })

# ---- Main Entrypoint ----
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
