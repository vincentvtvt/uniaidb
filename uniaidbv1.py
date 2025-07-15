import os
import json
import logging
import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

# ENV CONFIG (fail if not set)
DB_URL = os.environ['DATABASE_URL']
CLAUDE_API_KEY = os.environ['CLAUDE_API_KEY']
WASSENGER_API_KEY = os.environ['WASSENGER_API_KEY']

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
WASSENGER_API_URL = "https://api.wassenger.com/v1/messages"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uniai-prod")

app = FastAPI()


class MessageIn(BaseModel):
    bot_phone: str
    from_number: str
    message: str


# DB HELPERS

async def get_db():
    return await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)


async def get_customer(conn, phone_number) -> int:
    row = await conn.fetchrow("SELECT id FROM customers WHERE phone_number=$1", phone_number)
    if row:
        return row['id']
    row = await conn.fetchrow(
        "INSERT INTO customers (phone_number) VALUES ($1) RETURNING id", phone_number)
    return row['id']


async def get_bot(conn, phone_number) -> dict:
    row = await conn.fetchrow(
        "SELECT * FROM bots WHERE phone_number=$1", phone_number)
    if not row:
        raise Exception("Bot not found")
    return dict(row)


async def get_open_session(conn, bot_id, customer_id) -> Optional[int]:
    row = await conn.fetchrow(
        "SELECT id FROM sessions WHERE bot_id=$1 AND customer_id=$2 AND status='open' ORDER BY started_at DESC LIMIT 1",
        bot_id, customer_id)
    if row:
        return row['id']
    row = await conn.fetchrow(
        """INSERT INTO sessions (bot_id, customer_id, status) VALUES ($1, $2, 'open') RETURNING id""",
        bot_id, customer_id)
    return row['id']


async def save_message(conn, session_id, sender_type, sender_id, msg_text, meta=None, payload=None):
    await conn.execute("""
        INSERT INTO messages (session_id, sender_type, sender_id, message, meta, payload)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, session_id, sender_type, sender_id, msg_text, json.dumps(meta or {}), json.dumps(payload or {}))


async def get_tools(conn, bot_id):
    records = await conn.fetch("""
        SELECT t.*
        FROM bot_tools bt
        JOIN tools t ON bt.tool_id = t.id
        WHERE bt.bot_id=$1 AND bt.active AND t.active
    """, bot_id)
    return [dict(r) for r in records]


def build_personality_prompt(bot, tools, system_prompt, tool_prompt=None):
    # Combine bot system_prompt + tools + tool prompt if present
    tools_desc = "\n".join([f"{t['tool_id']}: {t['description']}" for t in tools])
    return f"{system_prompt}\n\nAvailable tools:\n{tools_desc}\n{tool_prompt or ''}".strip()


# AI CALLS

async def call_claude(prompt, history: Optional[List[dict]] = None, max_tokens: int = 1024):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    # History should be: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
    payload = {
        "model": "claude-3-haiku-20240307",  # Or another model if preferred
        "max_tokens": max_tokens,
        "messages": history or [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
        return result['content'][0]['text'] if isinstance(result['content'], list) else result['content']


async def ai_split_reply(full_reply: str, split_hint: Optional[str] = None) -> List[str]:
    """
    Use Claude to propose split points for 2-3 WhatsApp messages.
    """
    if len(full_reply) < 700:
        return [full_reply]
    prompt = (
        "Please split the following WhatsApp reply into 2 or 3 natural, human-sounding parts (messages), "
        "each under 700 characters if possible. Avoid splitting in the middle of a sentence. "
        "Return as a JSON array of strings.\n\n"
        f"Text:\n{full_reply}"
    )
    split_resp = await call_claude(prompt, max_tokens=1024)
    # Try to parse as JSON array
    try:
        arr = json.loads(split_resp)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except Exception:
        pass
    # fallback: split by sentences to ~700 chars
    out, current = [], ""
    for sent in full_reply.split(". "):
        if len(current) + len(sent) > 680 and current:
            out.append(current.strip())
            current = ""
        current += sent + ". "
    if current:
        out.append(current.strip())
    return out


# WhatsApp sending
async def send_whatsapp(phone: str, text: str, device: str):
    headers = {
        "Token": WASSENGER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "phone": phone,
        "message": text,
        "device": device
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(WASSENGER_API_URL, headers=headers, json=payload, timeout=30)
        if not (200 <= r.status_code < 300):
            logger.error(f"Wassenger send failed: {r.status_code} {r.text}")
            raise HTTPException(status_code=500, detail="Failed to send WhatsApp")
    logger.info(f"Sent WhatsApp to {phone}: {text[:60]}...")


#  WEBHOOK

@app.post("/webhook")
async def webhook(msg: MessageIn):
    async with await get_db() as pool:
        async with pool.acquire() as conn:
            # 1. Customer and Bot
            customer_id = await get_customer(conn, msg.from_number)
            bot = await get_bot(conn, msg.bot_phone)
            bot_id = bot['id']
            bot_cfg = bot['config'] or {}
            system_prompt = bot_cfg.get('system_prompt') if isinstance(bot_cfg, dict) else json.loads(bot_cfg).get('system_prompt', "You are a helpful, human-like assistant.")

            device_id = bot_cfg.get('device_id') if isinstance(bot_cfg, dict) else json.loads(bot_cfg).get('device_id')
            if not device_id:
                raise HTTPException(400, detail="Device ID missing in bot config")

            # 2. Session handling
            session_id = await get_open_session(conn, bot_id, customer_id)

            # 3. Log user message
            await save_message(conn, session_id, 'user', customer_id, msg.message, meta={"from": msg.from_number}, payload={})

            # 4. Tools
            tools = await get_tools(conn, bot_id)
            active_tools = [t for t in tools if t['active']]

            # 5. Tool/personality prompt
            # (Assume only one tool matches, or pick by logic if multiple)
            tool_prompt = ""
            if active_tools:
                # For now: Just use the first tool's prompt if present.
                for t in active_tools:
                    if t.get("prompt"):
                        tool_prompt = t["prompt"]
                        break
            personality = build_personality_prompt(bot, active_tools, system_prompt, tool_prompt)

            # 6. AI reply
            full_reply = await call_claude(f"{personality}\n\nUser: {msg.message}", max_tokens=800)

            # 7. Split for WhatsApp
            reply_chunks = await ai_split_reply(full_reply)
            logger.info(f"Split reply into {len(reply_chunks)} messages")

            # 8. Send each reply and log
            for chunk in reply_chunks:
                await send_whatsapp(msg.from_number, chunk, device_id)
                await save_message(conn, session_id, 'ai', bot_id, chunk, meta={"to": msg.from_number}, payload={})

            # 9. Done
            return {"status": "ok", "reply_chunks": reply_chunks}

# --- For local dev only ---
if __name__ == "____":
    import uvicorn
    uvicorn.run("uniaidbv1:app", port=9000, reload=True)
