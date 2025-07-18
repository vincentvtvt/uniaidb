import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_
from datetime import datetime

# --- Setup ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uniai")
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# === Models (simplified for demo, use your actual models) ===
class Bot(db.Model):
    __tablename__ = "bots"
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String)
    system_prompt = db.Column(db.Text)
    manager_system_prompt = db.Column(db.Text)
    config = db.Column(db.JSON)

class BotTool(db.Model):
    __tablename__ = "bot_tools"
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    tool_id = db.Column(db.Integer)
    active = db.Column(db.Boolean)

class Tool(db.Model):
    __tablename__ = "tools"
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.String)  # or Integer, match your DB
    name = db.Column(db.String)
    prompt = db.Column(db.Text)
    active = db.Column(db.Boolean)

# === Wassenger Send Stub ===
def send_whatsapp_message(phone, message):
    log.info(f"[WASSENGER] Would send to {phone}: {message}")
    # Your real API call here

# === Main Webhook Handler ===
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        log.info(f"Webhook Received: {data}")

        # Extract phone (adjust per your inbound format!)
        phone = data.get("phone") or data.get("from") or data.get("sender", {}).get("phone")
        msg_text = data.get("message") or data.get("body") or data.get("text")

        log.info(f"Incoming from: {phone}, message: {msg_text}")

        # 1. Find Bot by phone_number
        bot = Bot.query.filter_by(phone_number=phone).first()
        if not bot:
            log.error("Bot not found for phone_number: %s", phone)
            return jsonify({"error": "Bot not found"}), 400
        log.info(f"Bot loaded: {bot.id}, system_prompt: {bot.system_prompt}")

        # 2. Find active default tool for this bot
        default_tool = (
            db.session.query(Tool)
            .join(BotTool, Tool.id == BotTool.tool_id)
            .filter(and_(BotTool.bot_id == bot.id, Tool.active == True, BotTool.active == True))
            .order_by(Tool.id.asc())
            .first()
        )

        # Compose system prompt
        if default_tool:
            prompt = f"{bot.system_prompt}\n\n{default_tool.prompt}"
            log.info(f"Prompt used: {prompt}")
        else:
            prompt = bot.system_prompt
            log.info("No active default tool, using only system_prompt")

        # 3. (Stub) AI reply (replace with your actual AI logic)
        ai_reply = f"AI reply to '{msg_text}' using prompt: {prompt[:40]}..."

        log.info(f"AI reply generated: {ai_reply}")

        # 4. Send via Wassenger (stubbed)
        send_whatsapp_message(phone, ai_reply)

        # 5. Return status
        return jsonify({"status": "ok", "ai_reply": ai_reply}), 200

    except Exception as e:
        log.exception("Error in webhook")
        return jsonify({"error": str(e)}), 500

# === Main Entrypoint ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
