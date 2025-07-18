import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, desc
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

# === Anthropic Claude API Client (Streaming) ===
claude_api_key = os.getenv("CLAUDE_API_KEY")
claude_model = os.getenv("LAUDE_MODEL", "claude-3-7-sonnet-20250219")
anthropic_client = anthropic.Anthropic(api_key=claude_api_key)

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

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    customer_phone = db.Column(db.String(30))
    session_id = db.Column(db.String(50))
    direction = db.Column(db.String(10))  # 'in'/'out'
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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

def find_tool_by_id(tools, tool_id):
    for tool in tools:
        if tool.tool_id == tool_id:
            return tool
    return None

def get_recent_messages(bot_id, customer_phone, session_id, limit=20):
    messages = (
        Message.query.filter_by(bot_id=bot_id, customer_phone=customer_phone, session_id=session_id)
        .order_by(desc(Message.created_at))
        .limit(limit)
        .all()
    )
    return list(reversed(messages))  # from oldest to newest

def save_message(bot_id, customer_phone, session_id, direction, content):
    msg = Message(
        bot_id=bot_id,
        customer_phone=customer_phone,
        session_id=session_id,
        direction=direction,
        content=content,
        created_at=datetime.utcnow()
    )
    db.session.add(msg)
    db.session.commit()
    logger.info(f"[DB] Saved message: {direction} - {content}")

def start_or_get_session_id(msg):
    # For now, session_id = customer_phone (simple); can use real session logic if needed
    return msg["fromNumber"]

# === Claude Streaming Utility ===
def stream_claude_reply(system_prompt, messages):
    logger.info("[Claude] Streaming Claude response...")
    stream = anthropic_client.messages.create(
        model=claude_model,
        max_tokens=8192,
        system=system_prompt,
        messages=messages,
        stream=True
    )
    output = ""
    for event in stream:
        delta = getattr(event, "delta", None)
        if delta and hasattr(delta, "text"):
            output += delta.text
            print(delta.text, end="", flush=True)  # Print to console live
            logger.info(f"[Claude-Stream] {delta.text}")
    return output

# === Step 1: Receive webhook, save message ===
@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    # Extract WhatsApp structure
    try:
        msg = data["data"]
        bot_phone = msg["toNumber"]
        user_phone = msg["fromNumber"]
        device_id = data["device"]["id"]
        msg_text = msg["body"]
    except Exception as e:
        logger.error(f"[WEBHOOK] Invalid incoming data: {e}")
        return jsonify({"error": "Invalid request format"}), 400

    # Lookup Bot
    bot = get_bot_by_phone(bot_phone)
    if not bot:
        logger.error(f"[ERROR] No bot found for phone {bot_phone}")
        return jsonify({"error": "Bot not found"}), 404

    session_id = start_or_get_session_id(msg)

    # 1. Save customer message (incoming)
    save_message(bot.id, user_phone, session_id, "in", msg_text)

    # 2. Retrieve last 20 messages for this session (context)
    context_msgs = get_recent_messages(bot.id, user_phone, session_id, limit=20)

    # Format context for Claude (history, role mapping)
    claude_messages = []
    for m in context_msgs:
        role = "assistant" if m.direction == "out" else "user"
        claude_messages.append({"role": role, "content": m.content})

    # 3. Decide which tool to use (Claude with manager_system_prompt)
    tools = get_active_tools_for_bot(bot.id)
    tools_desc = "\n".join([f"{t.tool_id}|{t.description}" for t in tools])
    manager_prompt = (bot.manager_system_prompt or "") + f"\n<TOOLS>\n{tools_desc}\n</TOOLS>\n"
    logger.info(f"[MANAGER_PROMPT] {manager_prompt}")

    tool_selection = ""
    # Claude: ask which tool to use
    try:
        selection_stream = anthropic_client.messages.create(
            model=claude_model,
            max_tokens=256,
            system=manager_prompt,
            messages=claude_messages + [{"role": "user", "content": msg_text}],
            stream=True
        )
        tool_reply = ""
        for event in selection_stream:
            delta = getattr(event, "delta", None)
            if delta and hasattr(delta, "text"):
                tool_reply += delta.text
                logger.info(f"[Manager-Stream] {delta.text}")
        tool_selection = tool_reply.strip()
        logger.info(f"[TOOL SELECTION RAW]: {tool_selection}")
        # Expected output: JSON or tool_id, parse it!
        import json
        if "tool_id" in tool_selection or "TOOLS" in tool_selection or "{" in tool_selection:
            selection_json = json.loads(tool_selection.replace("'", "\""))
            selected_tool_id = selection_json.get("TOOLS") or selection_json.get("tool_id") or "defaultvpt"
        else:
            selected_tool_id = tool_selection
    except Exception as e:
        logger.error(f"[MANAGER] Tool selection error: {e}")
        selected_tool_id = "defaultvpt"

    # 4. Use tool's prompt on history conversation
    tool_obj = find_tool_by_id(tools, selected_tool_id) or find_tool_by_id(tools, "defaultvpt")
    tool_prompt = tool_obj.prompt if tool_obj else ""
    logger.info(f"[TOOL] Using tool: {selected_tool_id}, prompt: {tool_prompt}")

    # 5. Final prompt = system_prompt + tool prompt
    final_system_prompt = (bot.system_prompt or "") + "\n" + (tool_prompt or "")

    # 6. Claude: generate reply to customer (streaming)
    logger.info(f"[CLAUDE FINAL PROMPT] {final_system_prompt}")
    reply_text = stream_claude_reply(final_system_prompt, claude_messages + [{"role": "user", "content": msg_text}])

    # 7. Save assistant reply
    save_message(bot.id, user_phone, session_id, "out", reply_text)

    # 8. Send reply via Wassenger
    wassenger_status, wassenger_resp = send_wassenger_reply(user_phone, reply_text, device_id)

    # 9. Return API output
    return jsonify({
        "status": "ok",
        "tool": selected_tool_id,
        "reply": reply_text,
        "wassenger_status": wassenger_status,
        "wassenger_resp": wassenger_resp
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
