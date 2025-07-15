import os
import json
import logging
import asyncpg
from fastapi import FastAPI
from pydantic import BaseModel
import httpx  # For WhatsApp payload sending demo

# --- Config ---
DB_URL = os.environ.get("DATABASE_URL") or "postgresql://uniaidb_user:YOURPASSWORD@dpg-d1q7ef7fte5s73d1fplg-a.singapore-postgres.render.com/uniaidb"
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY") or "sk-..."
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "sk-..."

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ventopia")

app = FastAPI()

class MessageIn(BaseModel):
    bot_phone: str   # "+60108273799"
    from_number: str # customer phone
    message: str     # incoming message text

# --- DB Helper ---
async def get_db():
    return await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)

async def fetch_bot_config(conn, bot_phone):
    row = await conn.fetchrow("SELECT * FROM bots WHERE phone_number=$1 AND active", bot_phone)
    if not row:
        raise Exception("Bot not found or inactive!")
    return dict(row)

async def fetch_bot_tools(conn, bot_id):
    records = await conn.fetch("""
        SELECT t.*
        FROM bot_tools bt
        JOIN tools t ON bt.tool_id = t.id
        WHERE bt.bot_id=$1 AND bt.active AND t.active
    """, bot_id)
    return [dict(r) for r in records]

async def save_message(conn, session_id, sender_type, msg_text, meta=None, payload=None):
    await conn.execute("""
        INSERT INTO messages (session_id, sender_type, message, meta, payload)
        VALUES ($1, $2, $3, $4, $5)
    """, session_id, sender_type, msg_text, json.dumps(meta or {}), json.dumps(payload or {}))

# --- DYNAMIC TOOL FETCH ---
def get_default_tool(tools):
    for t in tools:
        if str(t.get("tool_id", "")).lower().startswith('default'):
            return t
    return None

def get_tool_by_id(tools, tool_id):
    for t in tools:
        if t.get("tool_id") == tool_id or str(t.get("id")) == str(tool_id):
            return t
    return None

# --- Claude decision (simulate) ---
async def call_claude_manager(prompt, history=None):
    # Simulate: should call your real manager Claude with prompt+history
    # Always returns Default for demo
    return {"TOOLS": "Default"}

# --- Tool action handler (simulate) ---
async def call_tool_action(tool, message, extra=None):
    if tool['action_type'] == "claude_sales":
        return "Hi! This is a sales reply from Claude."
    elif tool['action_type'] == "gpt_analyse":
        return "Here is your SWOT analysis (dummy data)."
    elif tool['action_type'] == "form_validate":
        return "Form validated."
    elif tool['action_type'] == "close_session":
        return "Session ended, no further action."
    else:
        return "Tool not supported."

# --- WhatsApp sending helper (edit as needed for your actual API) ---
async def send_whatsapp_message(phone, message, device):
    # Replace this with your actual API call (e.g., Wassenger, UltraMsg, etc.)
    # Here we just log for demo.
    logger.debug(f"Sending WhatsApp payload: {{'phone': '{phone}', 'message': '{message}', 'device': '{device}'}}")
    # Example: 
    # async with httpx.AsyncClient() as client:
    #     await client.post("https://api.wassenger.com/v1/messages", json={
    #         "phone": phone,
    #         "message": message,
    #         "device": device
    #     })

# --- Main Webhook ---
@app.post("/webhook")
async def webhook(msg: MessageIn):
    async with await get_db() as pool:
        async with pool.acquire() as conn:
            # 1. Fetch bot config
            bot = await fetch_bot_config(conn, msg.bot_phone)
            bot_id = bot['id']
            config = json.loads(bot['config'])
            logger.info(f"Loaded bot config: {config}")

            # 2. Fetch tools for this bot
            tools = await fetch_bot_tools(conn, bot_id)
            tool_map = {t['tool_id']: t for t in tools}

            # 3. Simulate session/customer lookup (or create one for demo)
            session_id = 1  # For demo

            # 4. Save user message
            await save_message(conn, session_id, 'user', msg.message, {"from": msg.from_number}, {})

            # 5. Build dynamic <TOOLS> list for Claude manager prompt (optional)
            tools_desc = "\n".join([f"{t['tool_id']}: {t['description']}" for t in tools])
            system_prompt = bot.get("system_prompt") or config.get("system_prompt") or "Default system prompt"
            manager_prompt = f"{system_prompt}\n\n<TOOLS>\n{tools_desc}\n</TOOLS>\nUser message: {msg.message}"

            # 6. Call Claude manager to get tool decision
            manager_decision = await call_claude_manager(manager_prompt)
            logger.info(f"Claude manager decision: {manager_decision}")

            # 7. Route to correct tool
            tool_chosen = manager_decision.get("TOOLS")
            ai_reply = None
            if tool_chosen == "Default":
                default_tool = get_default_tool(tools)
                if not default_tool:
                    ai_reply = "No default tool found for this bot."
                else:
                    ai_reply = await call_tool_action(default_tool, msg.message)
            elif tool_chosen in tool_map:
                ai_reply = await call_tool_action(tool_map[tool_chosen], msg.message)
            else:
                tool_by_id = get_tool_by_id(tools, tool_chosen)
                if tool_by_id:
                    ai_reply = await call_tool_action(tool_by_id, msg.message)
                else:
                    ai_reply = f"Tool '{tool_chosen}' not found for this bot."

            logger.info(f"AI reply: {ai_reply}")

            # 8. Save AI reply
            await save_message(conn, session_id, 'ai', ai_reply, {"to": msg.from_number}, {})

            # 9. Send WhatsApp message (use your actual integration here)
            device_id = bot.get("device_id", "")  # Make sure you have device_id
            await send_whatsapp_message(msg.from_number, ai_reply, device_id)

            # 10. Return real AI reply to user (API)
            return {"reply": ai_reply}

# --- For local dev: run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=9000, reload=True)
