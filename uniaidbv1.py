import os
import json
import logging
import asyncpg
from fastapi import FastAPI, Request
from pydantic import BaseModel

# --- Claude and OpenAI SDKs ---
import openai  # For GPT, image/audio, etc.

# --- Config (update accordingly) ---
DB_URL = os.environ.get("DATABASE_URL") or "postgresql://uniaidb_user:YOURPASSWORD@dpg-d1q7ef7fte5s73d1fplg-a.singapore-postgres.render.com/uniaidb"
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY") or "sk-..."  # Claude key if you use their SDK
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "sk-..."  # For GPT/image etc

# --- Logging ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ventopia")

# --- FastAPI setup ---
app = FastAPI()

# --- Message schema for inbound simulation ---
class MessageIn(BaseModel):
    bot_phone: str   # "+60108273799"
    from_number: str # customer phone
    message: str     # incoming message text

# --- DB Helper ---
async def get_db():
    return await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)

# --- Fetch bot config ---
async def fetch_bot_config(conn, bot_phone):
    row = await conn.fetchrow("SELECT * FROM bots WHERE phone_number=$1 AND active", bot_phone)
    if not row:
        raise Exception("Bot not found or inactive!")
    return dict(row)

# --- Fetch tools for bot ---
async def fetch_bot_tools(conn, bot_id):
    records = await conn.fetch("""
        SELECT t.*
        FROM bot_tools bt
        JOIN tools t ON bt.tool_id = t.id
        WHERE bt.bot_id=$1 AND bt.active AND t.active
    """, bot_id)
    return [dict(r) for r in records]

# --- Save message to DB ---
async def save_message(conn, session_id, sender_type, msg_text, meta=None, payload=None):
    await conn.execute("""
        INSERT INTO messages (session_id, sender_type, message, meta, payload)
        VALUES ($1, $2, $3, $4, $5)
    """, session_id, sender_type, msg_text, json.dumps(meta or {}), json.dumps(payload or {}))

# --- Claude decision (simulate) ---
async def call_claude_manager(prompt, history=None):
    # You'd use the real Claude SDK here.
    # For simulation, let's say Claude replies:
    # {"TOOLS": "Default"} or {"TOOLS": "MarketingSwot"}
    # Here we always return Default for demo
    return {"TOOLS": "Default"}

# --- Tool action handler (simulate) ---
async def call_tool_action(tool, message, extra=None):
    # Depending on tool['action_type'] call Claude or OpenAI
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
            tool_map = {t['tool_id'] if 'tool_id' in t else t['id']: t for t in tools}

            # 3. Simulate session/customer lookup (or create one for demo)
            session_id = 1  # For demo, use 1

            # 4. Save user message
            await save_message(conn, session_id, 'user', msg.message, {"from": msg.from_number}, {})

            # 5. Compose system prompt for Claude
            system_prompt = bot.get("system_prompt") or config.get("system_prompt") or "Default system prompt"
            manager_decision = await call_claude_manager(system_prompt + "\n\n" + msg.message)
            logger.info(f"Claude manager decision: {manager_decision}")

            # 6. If Default, reply normally
            if manager_decision.get("TOOLS") == "Default":
                ai_reply = await call_tool_action(tool_map['defaultvpt'], msg.message)
            else:
                tool_name = manager_decision["TOOLS"]
                # find tool by string ID or integer
                tool = None
                for t in tools:
                    if t.get("tool_id") == tool_name or t.get("name") == tool_name or t.get("id") == tool_name:
                        tool = t
                        break
                if not tool:
                    ai_reply = f"Tool '{tool_name}' not found for this bot."
                else:
                    ai_reply = await call_tool_action(tool, msg.message)

            # 7. Save AI reply
            await save_message(conn, session_id, 'ai', ai_reply, {"to": msg.from_number}, {})

            logger.info(f"AI reply: {ai_reply}")

            # 8. Return
            return {"reply": ai_reply}

# --- For local dev: run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=9000, reload=True)
