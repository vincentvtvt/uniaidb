import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime
import requests

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "postgresql://...")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# === Database Models ===
class Bot(db.Model):
    __tablename__ = 'bots'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    phone_number = db.Column(db.String)
    config = db.Column(db.JSON)
    created_at = db.Column(db.DateTime)
    system_prompt = db.Column(db.Text)
    manager_system_prompt = db.Column(db.Text)

class Tool(db.Model):
    __tablename__ = 'tools'
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.String)
    name = db.Column(db.String)
    description = db.Column(db.Text)
    prompt = db.Column(db.Text)
    action_type = db.Column(db.String)
    active = db.Column(db.Boolean)

class BotTool(db.Model):
    __tablename__ = 'bot_tools'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    tool_id = db.Column(db.Integer)
    active = db.Column(db.Boolean)

# === WhatsApp Sending Function ===
def send_whatsapp_reply(to, text):
    # Example: Wassenger API. Replace as needed.
    # url = "https://api.wassenger.com/v1/messages"
    # headers = {"Content-Type": "application/json", "Token": os.getenv("WASSENGER_API_KEY")}
    # payload = {"phone": to, "message": text}
    # resp = requests.post(url, json=payload, headers=headers)
    # logging.info(f"Wassenger API response: {resp.status_code} {resp.text}")
    logging.info(f"(SIMULATED SEND) To: {to}, Message: {text}")

# === Webhook Endpoint ===
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        logging.info("==== Incoming webhook request ====")
        data = request.json
        logging.info(f"Request body: {data}")

        # (Customize parsing as per Wassenger payload!)
        msg_text = data.get('message', {}).get('body')
        wa_phone = data.get('message', {}).get('from')  # WhatsApp sender phone
        bot_phone = data.get('device', {}).get('phone')
        logging.info(f"msg_text: {msg_text}, wa_phone: {wa_phone}, bot_phone: {bot_phone}")

        if not bot_phone:
            return jsonify({"error": "Bot phone not found in payload"}), 400

        # === 1. Find the correct bot by phone_number ===
        bot = Bot.query.filter_by(phone_number=bot_phone).first()
        logging.info(f"Loaded bot: {bot}")

        if not bot:
            return jsonify({"error": "No bot matched this phone number"}), 404

        system_prompt = bot.system_prompt or ""
        logging.info(f"system_prompt: {system_prompt}")

        # === 2. Find active tools for this bot ===
        bot_tools = BotTool.query.filter(and_(
            BotTool.bot_id == bot.id,
            BotTool.active == True
        )).all()
        tool_ids = [bt.tool_id for bt in bot_tools]
        logging.info(f"tool_ids for this bot: {tool_ids}")

        if tool_ids:
            tools = Tool.query.filter(and_(
                Tool.id.in_(tool_ids),
                Tool.active == True
            )).all()
        else:
            tools = []

        # === 3. Compose response prompt ===
        response_prompt = system_prompt.strip()
        if tools:
            response_prompt += "\n\n"
            for tool in tools:
                response_prompt += f"[{tool.name}]: {tool.prompt}\n"
            logging.info("Composed response_prompt with tools")
        else:
            logging.info("No active tools linked to this bot, using system_prompt only")

        logging.info(f"Final response_prompt:\n{response_prompt}")

        # === 4. Simulate AI or call external API here ===
        ai_reply = f"AI (simulated) would answer: {msg_text} (prompt used: {response_prompt})"
        logging.info(f"ai_reply: {ai_reply}")

        # === 5. Send WhatsApp reply (real implementation needed) ===
        send_whatsapp_reply(wa_phone, ai_reply)

        return jsonify({"status": "ok", "reply": ai_reply}), 200

    except Exception as e:
        logging.exception("Webhook processing failed")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
