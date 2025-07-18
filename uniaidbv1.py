import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime
import requests
import anthropic

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniAI")

# === Flask & SQLAlchemy Config ===
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# === Claude API Client (Anthropic) ===
claude_api_key = os.getenv("CLAUDE_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=claude_api_key)

def get_claude_reply(system_prompt, user_message):
    logger.info("[Claude] Prompt to Claude:\n%s", system_prompt)
    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    text = response.content[0].text
    logger.info("[Claude] Response: %s", text)
    return text

# === Models ===
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
    tool_id = db.Column(db.String(50))
    active = db.Column(db.Boolean)

# === Wassenger Send Message ===
def send_wassenger_reply(phone, text, device_id):
    logger.info(f"[WASSENGER] To: {phone} | Device: {device_id} | Text: {text}")
    url = "https://api.wassenger.com/v1/messages"
    headers = {
        "Token": os.getenv("WASSENGER_API_KEY"),
        "Content-Type": "application/json"
    }
    payload = {
        "phone": phone,
        "message": text,
        "device": device_id
    }
    resp = requests.post(url, json=payload, headers=headers)
    logger.info(f"[WASSENGER] Status: {resp.status_code} | Response: {resp.text}")
    return resp.status_code, resp.text

# === DB Utility Functions ===
def get_bot_by_phone(phone_number):
    bot = Bot.query.filter_by(phone_number=phone_number).first()
    logger.info(f"[DB] Bot lookup by phone: {phone_number} => {bot}")
    return bot

def get_active_tools_for_bot(bot_id):
    tools = (
        db.session.query(Tool)
        .join(BotTool, and_(
            Tool.tool_id == BotTool.tool_id,
            BotTool.bot_id == bot_id,
            Tool.active == True,
            BotTool.active == True
        ))
        .all()
    )
    logger.info(f"[DB] Tools for bot_id={bot_id}: {[t.tool_id for t in tools]}")
    return tools

def find_default_tool(tools):
    for tool in tools:
        if tool.tool_id == "defaultvpt":
            return tool
    return None

# === Main Webhook Handler ===
@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    # Wassenger structure extraction
    try:
        msg = data["data"]
        bot_phone = msg["toNumber"]
        user_phone = msg["fromNumber"]
        device_id = data["device"]["id"]
        msg_text = msg["body"]
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # 1. Find bot by phone number
    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404

    # 2. Get all active tools for the bot
    tools = get_active_tools_for_bot(bot.id)
    default_tool = find_default_tool(tools)

    # 3. Assemble prompt (system_prompt + tool prompt)
    if default_tool:
        final_prompt = (bot.system_prompt or "") + "\n" + (default_tool.prompt or "")
        logger.info(f"[PROMPT] Using system_prompt + defaultvpt: {final_prompt}")
    else:
        final_prompt = bot.system_prompt or ""
        logger.warning("[PROMPT] No defaultvpt tool found, using system_prompt only")
    logger.info(f"[PROMPT] Final prompt: {final_prompt}")

    # 4. Call Claude to generate reply
    ai_reply = get_claude_reply(final_prompt, msg_text)
    logger.info(f"[Claude] AI reply: {ai_reply}")

    # 5. Send reply via Wassenger
    wassenger_status, wassenger_resp = send_wassenger_reply(user_phone, ai_reply, device_id)

    # 6. Log and respond
    return jsonify({
        "status": "ok",
        "used_prompt": final_prompt,
        "ai_reply": ai_reply,
        "wassenger_status": wassenger_status,
        "wassenger_resp": wassenger_resp
    })

# === Flask run (dev only) ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
