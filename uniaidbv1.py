import os
import json
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
import anthropic

DATABASE_URL = os.getenv("DATABASE_URL")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")

app = Flask(__name__)
Base = declarative_base()

# --- MODELS ---
class Tool(Base):
    __tablename__ = 'tools'
    id = Column(Integer, primary_key=True)
    tool_id = Column(String(50), unique=True)
    name = Column(String(100))
    description = Column(Text)
    prompt = Column(Text)
    active = Column(Boolean, default=True)

class Bot(Base):
    __tablename__ = 'bots'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    phone_number = Column(String(30), unique=True)
    manager_system_prompt = Column(Text)  # for tool selection
    system_prompt = Column(Text)          # for default flow (customer reply)
    config = Column(Text)                 # for other config if needed

class BotTools(Base):
    __tablename__ = 'bot_tools'
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer)
    tool_id = Column(String(50))
    active = Column(Boolean, default=True)

# --- DB Setup ---
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine))

# --- Helpers (same as above, not shown for brevity) ---
# ... [get_tools_table_prompt, build_manager_prompt, call_claude, orchestration_logic, etc. as in earlier code]

# (Paste your orchestration_logic, get_tools_table_prompt, etc. here, unchanged!)

def get_tools_table_prompt(bot_id):
    tool_links = db_session.query(BotTools).filter(
        BotTools.bot_id == bot_id, BotTools.active == True
    ).all()
    tool_ids = [t.tool_id for t in tool_links]
    tools = db_session.query(Tool).filter(
        Tool.tool_id.in_(tool_ids), Tool.active == True
    ).all()
    tools = sorted(tools, key=lambda t: 0 if t.tool_id == "Default" else 1)
    table = ""
    for t in tools:
        table += f"{t.tool_id}|{t.description}\n"
    return table.strip()

def build_manager_prompt(manager_system_prompt, tools_table):
    return f"""{manager_system_prompt}

<TOOLS>
{tools_table}
</TOOLS>

<ExampleOutput>
{{
  "TOOLS": "Default"
}}
</ExampleOutput>

IMPORTANT: Only output a pure JSON object matching the ExampleOutput, with the selected TOOL ID. No greetings, no explanations, no extra text—just the JSON object.
"""

def call_claude(prompt, user_message, max_tokens=256):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        system=prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text if message and message.content else ""

def orchestration_logic(user_message, bot_id):
    bot = db_session.query(Bot).filter(Bot.id == bot_id).first()
    if not bot:
        return {"error": "Invalid bot_id"}, 400

    # --- Stage 1: Tool Selection ---
    tools_table = get_tools_table_prompt(bot_id)
    if "Default" not in tools_table:
        return {"error": "No Default tool linked to this bot"}, 500
    manager_prompt = build_manager_prompt(bot.manager_system_prompt, tools_table)
    print("[Manager System Prompt]:", manager_prompt)
    print("[User Message]:", user_message)
    ai_result_raw = call_claude(manager_prompt, user_message)
    print("[Manager Claude Raw Output]:", ai_result_raw)
    try:
        ai_result = json.loads(ai_result_raw)
        print("[Manager AI JSON]:", ai_result)
    except Exception as e:
        print(f"[Manager Claude JSON error]: {e} | Output: {ai_result_raw}")
        return {"error": "Manager Claude returned invalid JSON", "raw": ai_result_raw}, 500

    tool_id = ai_result.get("TOOLS")
    if not tool_id:
        return {"error": "No TOOL selected by manager Claude", "raw": ai_result_raw}, 500

    # --- Stage 2: Route Based on Tool ---
    if tool_id == "Default":
        # Proceed to customer-facing agent, use system_prompt for reply
        print("[Routing] Manager chose Default; calling customer agent.")
        customer_prompt = bot.system_prompt
        ai_cust_raw = call_claude(customer_prompt, user_message, max_tokens=512)
        print("[Customer Agent Raw Output]:", ai_cust_raw)
        try:
            ai_cust = json.loads(ai_cust_raw)
            print("[Customer Agent AI JSON]:", ai_cust)
        except Exception as e:
            print(f"[Customer Agent JSON error]: {e} | Output: {ai_cust_raw}")
            ai_cust = {}
        return {"stage": "default", "customer_agent_reply": ai_cust, "raw": ai_cust_raw}, 200

    else:
        # Use the selected tool's prompt
        tool = db_session.query(Tool).filter(Tool.tool_id == tool_id).first()
        tool_prompt = tool.prompt if tool and tool.prompt else f"你是一个工具助手，请执行 {tool_id} 对应的操作，输出规范JSON。"
        print(f"[Routing] Manager chose tool {tool_id}; calling tool agent.")
        ai_tool_raw = call_claude(tool_prompt, user_message, max_tokens=256)
        print(f"[Tool Agent Raw Output for {tool_id}]:", ai_tool_raw)
        try:
            ai_tool = json.loads(ai_tool_raw)
            print(f"[Tool Agent AI JSON for {tool_id}]:", ai_tool)
        except Exception as e:
            print(f"[Tool Agent JSON error for {tool_id}]: {e} | Output: {ai_tool_raw}")
            ai_tool = {}
        return {"stage": "tool", "tool_id": tool_id, "tool_agent_reply": ai_tool, "raw": ai_tool_raw}, 200

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Universal webhook for Wassenger format. Extracts text, bot number, and calls orchestration.
    """
    payload = request.get_json(force=True)
    print("[Wassenger Payload]:", json.dumps(payload, ensure_ascii=False, indent=2))

    # Check it's a new inbound message event
    if not payload or payload.get("event") != "message:in:new":
        return jsonify({"ignored": True}), 200

    data = payload.get("data", {})
    msg_text = data.get("text") or data.get("body") or ""  # handle text or body
    bot_phone = (data.get("to") or "").replace("+", "").replace("@c.us", "")  # normalize
    if not msg_text or not bot_phone:
        return jsonify({"error": "Missing text or bot phone"}), 400

    # Lookup bot_id by bot_phone (try with/without "+")
    bot = db_session.query(Bot).filter(
        (Bot.phone_number == bot_phone) | 
        (Bot.phone_number == f"+{bot_phone}")
    ).first()
    if not bot:
        print(f"[Webhook] No bot found for phone: {bot_phone}")
        return jsonify({"error": "Bot not found for phone", "phone": bot_phone}), 404

    bot_id = bot.id

    # Pass to orchestration
    result, status = orchestration_logic(msg_text, bot_id)
    # Optionally: Send AI result back to WhatsApp here via Wassenger send API

    return jsonify(result), status

@app.route('/ai-universal', methods=['POST'])
def ai_universal():
    data = request.get_json(force=True)
    user_message = data.get('message')
    bot_id = data.get('bot_id')
    if not user_message or not bot_id:
        return jsonify({'error': 'Missing user_message or bot_id'}), 400
    result, status = orchestration_logic(user_message, bot_id)
    return jsonify(result), status

@app.route('/')
def home():
    return "Universal AI Bot is running.", 200

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 18080))
    app.run(host='0.0.0.0', port=port, debug=True)
