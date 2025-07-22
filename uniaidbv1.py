import os
import re
import logging
import time
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import requests
import openai
import base64
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
import cv2




# --- Universal JSON Prompt Builder ---
def build_json_prompt(base_prompt, example_json, tag=None):
    tag_name = tag if tag else "ExampleOutput"
    json_instruction = (
        "\n\n<OutputFormat>\n"
        "Always respond ONLY with a strict, valid JSON object. "
        "Use double quotes for all keys and string values. "
        "Do not include any explanation, markdown, or code block formatting—just pure JSON.\n"
        f"Wrap your response inside <{tag_name}> tags as shown below.\n"
        "</OutputFormat>\n"
        f"<{tag_name}>\n"
        f"{example_json.strip()}\n"
        f"</{tag_name}>"
    )
    return base_prompt.strip() + json_instruction

def strip_json_markdown_blocks(text):
    """Removes ```json ... ``` or ``` ... ``` wrappers from AI output."""
    return re.sub(r'```[a-z]*\s*([\s\S]*?)```', r'\1', text, flags=re.MULTILINE).strip()

def build_json_prompt_with_reasoning(base_prompt, example_json, tag=None):
    tag_name = tag if tag else "ExampleOutput"
    reasoning_instruction = (
        "\n\n<Reasoning>\n"
        "Before answering, briefly explain your reasoning for the tool selection in 1-2 sentences."
        " After your reasoning, output ONLY the strict JSON inside <{tag}> tags as shown below."
        " Do not add code block formatting or any other explanation after the JSON.\n"
        "</Reasoning>"
    ).replace("{tag}", tag_name)
    json_instruction = (
        reasoning_instruction +
        "\n\n<OutputFormat>\n"
        "Always respond ONLY with a strict, valid JSON object. "
        "Use double quotes for all keys and string values. "
        "No markdown, no code block formatting.\n"
        f"Wrap your response inside <{tag_name}> tags as shown below.\n"
        "</OutputFormat>\n"
        f"<{tag_name}>\n"
        f"{example_json.strip()}\n"
        f"</{tag_name}>"
    )
    return base_prompt.strip() + json_instruction




# Example usage (manager decision):
# manager_prompt = build_json_prompt(bot.manager_system_prompt, '{\n  "TOOLS": "Default"\n}', tag="ExampleOutput")

# Example usage (reply/tool generation):
# reply_prompt = build_json_prompt(bot.system_prompt, '{\n  "message": ["hello", "world"]\n}', tag="ExampleOutput")


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("UniAI")

openai.api_key = os.getenv("OPENAI_API_KEY")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Models ---
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
    customer_phone = db.Column(db.String(50))
    session_id = db.Column(db.String(50))
    direction = db.Column(db.String(10))  # 'in' or 'out'
    content = db.Column(db.Text)
    raw_media_url = db.Column(db.Text)
    created_at = db.Column(db.DateTime)

class Customer(db.Model):
    __tablename__ = 'customers'
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100))
    language = db.Column(db.String(10))
    meta = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Session(db.Model):
    __tablename__ = 'session'
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'))
    bot_id = db.Column(db.Integer)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='open')  # 'open' or 'closed'
    context = db.Column(db.JSON, default={})
    customer = db.relationship("Customer")

class Template(db.Model):
    __tablename__ = 'templates'
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.String(50))
    bot_id = db.Column(db.Integer)
    description = db.Column(db.String(255))
    content = db.Column(db.JSON)  # JSONB
    language = db.Column(db.String(10))
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)

# --- Helpers ---
def download_file(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def encode_image_b64(img_bytes):
    return base64.b64encode(img_bytes).decode()

def extract_text_from_image(img_url, prompt=None):
    image_bytes = download_wassenger_media(img_url)
    img_b64 = encode_image_b64(image_bytes)
    logger.info("[VISION] Sending image to OpenAI Vision...")
    messages = [
        {"role": "system", "content": prompt or "Extract all visible text from this image. If no text, describe what you see. Example output is 'user is sending a image with chicken eating chicken'"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
    ]
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=8192
    )
    return result.choices[0].message.content.strip()

def transcribe_audio_from_url(audio_url):
    # Use Wassenger media download for correct auth
    audio_bytes = download_wassenger_media(audio_url)
    if not audio_bytes or len(audio_bytes) < 1024:
        logger.error("[AUDIO DOWNLOAD] Failed or too small")
        return "[audio received, transcription failed]"
    temp_path = "/tmp/temp_audio.ogg"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        logger.info(f"[WHISPER] Transcript: {transcript.text.strip()}")
        return transcript.text.strip()
    except Exception as e:
        logger.error(f"[WHISPER ERROR] {e}")
        return "[audio received, transcription failed]"


import re
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger("UniAI")

def extract_text_from_message(msg):
    import cv2
    import numpy as np
    from pdf2image import convert_from_bytes

    msg_type = msg.get("type")
    media = msg.get("media", {})
    msg_text, media_url = None, None

    # Helper: get downloadable media URL (Wassenger)
    def get_media_url(media):
        url = media.get("url")
        if not url and "links" in media and "download" in media["links"]:
            url = "https://api.wassenger.com" + media["links"]["download"]
        return url

    # Helper: extract first frame from video (OpenCV)
    def extract_first_frame_from_video(video_bytes):
        try:
            np_arr = np.frombuffer(video_bytes, np.uint8)
            video_file = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if video_file is None:
                temp_path = "/tmp/temp_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(video_bytes)
                cap = cv2.VideoCapture(temp_path)
            else:
                cap = cv2.VideoCapture(video_file)
            success, frame = cap.read()
            if success and frame is not None:
                _, buf = cv2.imencode('.png', frame)
                return buf.tobytes()
        except Exception as e:
            logger.error(f"[VIDEO FRAME] Failed: {e}")
        return None

    # --- Sticker ---
    if msg_type == "sticker":
        img_url = get_media_url(media)
        logger.info(f"[STICKER DEBUG] img_url for Vision: {img_url}")
        if img_url:
            try:
                image_bytes = download_wassenger_media(img_url)
                # Convert .webp to .png
                if media.get("extension", "").lower() == "webp":
                    im = Image.open(BytesIO(image_bytes)).convert("RGBA")
                    buf = BytesIO()
                    im.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                img_b64 = encode_image_b64(image_bytes)
                vision_msg = [
                    {"role": "system", "content": (
                        "This is a WhatsApp sticker. "
                        "Briefly describe what is shown in the sticker, focusing on the main character, action, and emotion. "
                        "Example output is 'user is sending a sticker with chicken eating chicken'"
                        "If there is text in the sticker, include it. "
                        "Reply in a short, natural phrase, no code formatting."
                    )},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]}
                ]
                result = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=vision_msg,
                    max_tokens=8192
                )
                msg_text = result.choices[0].message.content.strip()
                return msg_text or "[Sticker received]", img_url
            except Exception as e:
                logger.error(f"[STICKER MEANING] {e}")
                return "[Sticker received]", img_url
        return "[Sticker received]", None

    # --- Image ---
    elif msg_type == "image":
        img_url = get_media_url(media)
        logger.info(f"[IMAGE DEBUG] img_url for Vision: {img_url}")
        if img_url:
            try:
                image_bytes = download_wassenger_media(img_url)
                img_b64 = encode_image_b64(image_bytes)
                vision_msg = [
                    {"role": "system", "content": (
                        "This is a photo/image received on WhatsApp. "
                        "Summarize briefly what you see, focusing on main objects, scene, or text. "
                        "Example output is 'user is sending a image with chicken eating chicken'"
                        "Reply in a short phrase. If there is text, mention it."
                    )},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]}
                ]
                result = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=vision_msg,
                    max_tokens=8192
                )
                msg_text = result.choices[0].message.content.strip()
                return msg_text or "[Image received]", img_url
            except Exception as e:
                logger.error(f"[IMAGE MEANING] {e}")
                return "[Image received]", img_url
        return "[Image received, no url]", None

    # --- Video (first frame to Vision) ---
    elif msg_type == "video":
        vid_url = get_media_url(media)
        file_name = media.get("filename") or ""
        if vid_url:
            try:
                video_bytes = download_wassenger_media(vid_url)
                frame_bytes = extract_first_frame_from_video(video_bytes)
                if frame_bytes:
                    img_b64 = encode_image_b64(frame_bytes)
                    vision_msg = [
                        {"role": "system", "content": (
                            "This is the first frame of a WhatsApp video. "
                            "Summarize the scene—main subject, action, or text. Short phrase."
                            "Example output is 'user is sending a video with chicken eating chicken'"
                        )},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]}
                    ]
                    result = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=vision_msg,
                        max_tokens=128
                    )
                    msg_text = result.choices[0].message.content.strip()
                    return msg_text or f"[Video received: {file_name}]", vid_url
                else:
                    return f"[Video received: {file_name}]", vid_url
            except Exception as e:
                logger.error(f"[VIDEO MEANING] {e}")
                return f"[Video received: {file_name}]", vid_url
        return "[Video received, no url]", None

    # --- Audio (Whisper + GPT summary) ---
    elif msg_type == "audio":
        audio_url = get_media_url(media)
        if audio_url:
            try:
                transcript = transcribe_audio_from_url(audio_url)
                if transcript and transcript.lower() not in ("[audio received, no url]", "[audio received, transcription failed]"):
                    gpt_prompt = (
                        "This is a WhatsApp audio message transcribed as: "
                        f"'{transcript}'. Reply in a short, natural phrase, as if you're the user."
                    )
                    result = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": gpt_prompt}],
                        max_tokens=64
                    )
                    msg_text = result.choices[0].message.content.strip()
                    return {"transcript": transcript, "gpt_reply": msg_text}, audio_url
                else:
                    return {"transcript": transcript or "[Audio received, no speech detected]", "gpt_reply": None}, audio_url
            except Exception as e:
                logger.error(f"[AUDIO MEANING] {e}")
                return {"transcript": "[Audio received, error]", "gpt_reply": None}, audio_url
        return {"transcript": "[Audio received, no url]", "gpt_reply": None}, None


    # --- Document (PDF/image: Vision, else filename) ---
    elif msg_type == "document":
        doc_url = get_media_url(media)
        file_name = media.get("filename") or ""
        ext = (file_name.split(".")[-1] if file_name else "").lower()
        if doc_url:
            try:
                doc_bytes = download_wassenger_media(doc_url)
                if ext == "pdf":
                    images = convert_from_bytes(doc_bytes, first_page=1, last_page=1)
                    if images:
                        buf = BytesIO()
                        images[0].save(buf, format="PNG")
                        img_b64 = encode_image_b64(buf.getvalue())
                        vision_msg = [
                            {"role": "system", "content": (
                                "This is the first page of a PDF document sent via WhatsApp. "
                                "Example output is 'user is sending a document with details of abc'"
                                "Summarize what you see—any headings, tables, or visible text. Short natural phrase."
                            )},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                            ]}
                        ]
                        result = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=vision_msg,
                            max_tokens=128
                        )
                        msg_text = result.choices[0].message.content.strip()
                        return msg_text or f"[Document received: {file_name}]", doc_url
                elif ext in ("jpg", "jpeg", "png"):
                    img_b64 = encode_image_b64(doc_bytes)
                    vision_msg = [
                        {"role": "system", "content": (
                            "This is an image document received on WhatsApp. "
                            "Example output is 'user is sending a image with chicken eating chicken'"
                            "Summarize what you see—main subject, visible text. Short phrase."
                        )},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]}
                    ]
                    result = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=vision_msg,
                        max_tokens=128
                    )
                    msg_text = result.choices[0].message.content.strip()
                    return msg_text or f"[Document received: {file_name}]", doc_url
                else:
                    return f"[Document received: {file_name}]", doc_url
            except Exception as e:
                logger.error(f"[DOC MEANING] {e}")
                return f"[Document received: {file_name}]", doc_url
        return "[Document received, no url]", None

    # --- Fallback ---
    else:
        msg_text = msg.get("body") or msg.get("caption") or f"[{msg_type} received]" if msg_type else "[Message received]"
        return msg_text, None


def get_template_content(template_id):
    template = db.session.query(Template).filter_by(template_id=template_id, active=True).first()
    if not template or not template.content:
        return []
    return template.content if isinstance(template.content, list) else json.loads(template.content)

def upload_media_to_wassenger(img_url_or_bytes):
    """
    Uploads image to Wassenger via URL or bytes, returns file_id.
    Accepts either a direct image URL or raw bytes.
    """
    url = "https://api.wassenger.com/v1/files"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    # If string, treat as URL; else assume bytes and upload as base64
    if isinstance(img_url_or_bytes, str):
        payload = {"url": img_url_or_bytes}
    else:
        # For local bytes, convert to base64 and use payload {'data': ...}
        import base64
        payload = {"data": base64.b64encode(img_url_or_bytes).decode()}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        files = resp.json()
        if files and isinstance(files, list) and files[0].get('id'):
            return files[0]['id']
        else:
            logger.error(f"[MEDIA UPLOAD FAIL] Wassenger /files bad response: {resp.text}")
    except Exception as e:
        logger.error(f"[MEDIA UPLOAD FAIL] Wassenger /files error: {e}")
    return None

def download_wassenger_media(url):
    """
    Downloads a media file from Wassenger using API Key authentication.
    Returns bytes if successful, else None.
    """
    headers = {"Token": os.getenv("WASSENGER_API_KEY")}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.error(f"[WASSENGER MEDIA DOWNLOAD ERROR] {e}")
        return None

def upload_and_send_media(recipient, file_url_or_path, device_id, caption=None, msg_type=None, delay_seconds=5):
    """
    Universal helper: uploads image or PDF (from URL or file path), then sends via Wassenger.
    - recipient: phone in E164 ('+6012...') or group ('....@g.us')
    - file_url_or_path: remote URL or local path
    - device_id: Wassenger device ID
    - caption: WhatsApp message caption (optional)
    - msg_type: "image" or "media" (if None, auto-detect)
    """
    # 1. Upload file to Wassenger, get file_id
    filename = None
    # Guess msg_type if not provided
    if not msg_type:
        # Simple guess by extension
        if isinstance(file_url_or_path, str):
            if file_url_or_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                msg_type = "image"
            elif file_url_or_path.lower().endswith(('.pdf', '.doc', '.docx')):
                msg_type = "media"
            else:
                msg_type = "media"
        else:
            msg_type = "media"

    # Upload (if not already a file id)
    if isinstance(file_url_or_path, str) and len(file_url_or_path) == 24 and file_url_or_path.isalnum():
        file_id = file_url_or_path  # Already a file id
    else:
        # Download/upload
        file_id = upload_any_file_to_wassenger(file_url_or_path)

    if not file_id:
        logger.error(f"[UPLOAD & SEND] Failed to upload file for recipient {recipient}")
        return None

    # 2. Send via Wassenger
    return send_wassenger_reply(
        recipient,
        file_id,
        device_id,
        msg_type=msg_type,
        caption=caption,
        delay_seconds=delay_seconds
    )



def send_wassenger_reply(phone, text, device_id, delay_seconds=5, msg_type="text", caption=None):
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"device": device_id}

    # Recipient: phone or group
    if isinstance(phone, str) and phone.endswith("@g.us"):
        payload["group"] = phone
    else:
        payload["phone"] = phone

    if msg_type == "text":
        payload["message"] = text
        payload["schedule"] = {"delay": delay_seconds}

    elif msg_type == "image":
        # Use file ID if already uploaded, else try to upload or send by URL
        if isinstance(text, str) and len(text) == 24 and text.isalnum():
            payload["media"] = {"file": text}
        elif isinstance(text, str) and text.startswith("http"):
            payload["media"] = {"url": text}
        else:
            # Assume local file path or bytes, upload and use file ID
            file_id = upload_any_file_to_wassenger(text)
            if not file_id:
                logger.error("[SEND IMAGE] Failed to upload image to Wassenger")
                return
            payload["media"] = {"file": file_id}
        if caption:
            payload["message"] = caption

    elif msg_type == "media":  # PATCH: handle documents, pdfs, videos
        # Use file ID if already uploaded, else try to send by URL
        if isinstance(text, str) and len(text) == 24 and text.isalnum():
            payload["media"] = {"file": text}
        elif isinstance(text, str) and text.startswith("http"):
            payload["media"] = {"url": text}
        else:
            # Assume local file path or bytes, upload and use file ID
            file_id = upload_any_file_to_wassenger(text)
            if not file_id:
                logger.error("[SEND MEDIA] Failed to upload file to Wassenger")
                return
            payload["media"] = {"file": file_id}
        if caption:
            payload["message"] = caption

    else:
        logger.error(f"Unsupported msg_type: {msg_type}")
        return

    payload = {k: v for k, v in payload.items() if v is not None}
    logger.debug(f"[WASSENGER PAYLOAD]: {payload}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        logger.info(f"Wassenger response: {resp.status_code} {resp.text}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SEND WASSENGER ERROR] {e}")
        return None

def notify_sales_group(bot, message, error=False):
    import json
    config = bot.config
    if isinstance(config, str):
        config = json.loads(config)
    group_id = (config or {}).get("notification_group")
    device_id = (config or {}).get("device_id")
    if group_id and device_id:
        note = f"[ALERT] {message}" if error else message
        send_wassenger_reply(group_id, note, device_id)
    else:
        logger.warning("[NOTIFY] Notification group or device_id missing in bot.config")


def get_bot_by_phone(phone_number):
    num_variants = [
        phone_number,
        phone_number.lstrip('+'),
        phone_number.replace('+', '').replace('@c.us', ''),
        '+' + phone_number.replace('@c.us', '').lstrip('+'),
        phone_number.replace('@c.us', ''),
    ]
    logger.debug(f"[DB] Bot lookup attempt for: {phone_number} (variants: {num_variants})")
    for variant in num_variants:
        bot = Bot.query.filter_by(phone_number=variant).first()
        if bot:
            logger.debug(f"[DB] Bot found! Input: {phone_number}, Matched: {variant} (DB: {bot.phone_number})")
            return bot
    logger.error(f"[DB] Bot NOT FOUND for: {phone_number} (tried: {num_variants})")
    return None

def get_active_tools_for_bot(bot_id):
    tools = (
        db.session.query(Tool)
        .join(BotTool, (Tool.tool_id == BotTool.tool_id) & (BotTool.bot_id == bot_id) & (Tool.active == True) & (BotTool.active == True))
        .all()
    )
    logger.info(f"[DB] Tools for bot_id={bot_id}: {[t.tool_id for t in tools]}")
    return tools

def save_message(bot_id, customer_phone, session_id, direction, content, raw_media_url=None):
    msg = Message(
        bot_id=bot_id,
        customer_phone=customer_phone,
        session_id=session_id,
        direction=direction,
        content=content,
        raw_media_url=raw_media_url,
        created_at=datetime.now()
    )
    db.session.add(msg)
    db.session.commit()
    logger.info(f"[DB] Saved message ({direction}) for {customer_phone}: {content}")

def get_latest_history(bot_id, customer_phone, session_id, n=20):
    messages = (Message.query
        .filter_by(bot_id=bot_id, customer_phone=customer_phone, session_id=session_id)
        .order_by(Message.created_at.desc())
        .limit(n)
        .all())
    messages = messages[::-1]
    logger.info(f"[DB] History ({len(messages)} messages) loaded.")
    return messages

def build_tool_menu_for_prompt(bot_id):
    tools = get_active_tools_for_bot(bot_id)
    menu = []
    for t in tools:
        menu.append(f"{t.tool_id} ({t.name}): {t.description}")
    return "\n".join(menu)

def decide_tool_with_manager_prompt(bot, history):
    # Build history text as before
    history_text = "\n".join([f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}" for m in history])

    # ---- NEW: Build the tool menu and append to system prompt ----
    tool_menu = build_tool_menu_for_prompt(bot.id)
    tool_menu_text = (
        "Here are the available tools you can select (ID, name, and description):\n"
        f"{tool_menu}\n"
        "Choose the single most appropriate tool for this conversation."
    )

    # Merge with your manager prompt
    manager_prompt = build_json_prompt_with_reasoning(
        (bot.manager_system_prompt or "") + "\n" + tool_menu_text,
        '{\n  "TOOLS": "Default"\n}',
        tag="ExampleOutput"
    )
    logger.info(f"[AI DECISION] manager_system_prompt: {manager_prompt}")
    logger.info(f"[AI DECISION] history: {history_text}")
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": manager_prompt},
            {"role": "user", "content": history_text}
        ],
        max_tokens=8192,
        temperature=0
    )
    tool_decision = response.choices[0].message.content

    # Extract reasoning (for logging/printout)
    reasoning_match = re.search(r'<Reasoning>(.*?)</Reasoning>', tool_decision, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        logger.info(f"[AI TOOL DECISION REASONING]: {reasoning}")
        print("\n[AI TOOL DECISION REASONING]:", reasoning)
    else:
        logger.info("[AI TOOL DECISION REASONING]: None found")

    # Extract the JSON block inside <ExampleOutput>
    match = re.search(r'<ExampleOutput>(.*?)</ExampleOutput>', tool_decision, re.DOTALL)
    json_block = match.group(1).strip() if match else tool_decision
    match_json = re.search(r'"TOOLS":\s*"([^"]+)"', json_block)
    return match_json.group(1) if match_json else None


def compose_reply(bot, tool, history, context_input):
    # Compose reply using strict JSON format prompt
    if tool:
        prompt = (bot.system_prompt or "") + "\n" + (tool.prompt or "")
        example_json = '''{
  "template": "bf_UGnkL24bhtCQBIJr7hbT",
  "message": [
    "salam cik atas pakej astro fibre 2025",
    "cik ada astro tv yang aktif bulanan ke?"
  ]
}'''
    else:
        prompt = bot.system_prompt or ""
        example_json = '''{
  "message": [
    "astro fibre boleh sambung 10-20+ device",
    "astro go boleh stream 4 device",
    "nak proceed sila isi borang ye cik"
  ]
}'''
    reply_prompt = build_json_prompt(prompt, example_json, tag="ExampleOutput")
    logger.info(f"[AI REPLY] Prompt: {reply_prompt}")
    messages = [
        {"role": "system", "content": reply_prompt},
        {"role": "user", "content": context_input}
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=8192,
        temperature=0.3,
        stream=True
    )
    reply_accum = ""
    print("[STREAM] Streaming model reply:")
    for chunk in stream:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            reply_accum += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
    logger.info(f"\n[AI REPLY STREAMED]: {reply_accum}")
    # Extract JSON inside <ExampleOutput>
    match = re.search(r'<ExampleOutput>(.*?)</ExampleOutput>', reply_accum, re.DOTALL)
    return match.group(1).strip() if match else reply_accum

def process_ai_reply_and_send(customer_phone, ai_reply, device_id, bot_id=None, user=None, session_id=None):
    try:
        parsed = ai_reply if isinstance(ai_reply, dict) else json.loads(ai_reply)
    except Exception as e:
        logger.error(f"[WEBHOOK] Could not parse AI reply as JSON: {ai_reply} ({e})")
        parsed = {}

    # Defensive: Only continue if parsed is a dict
    if not isinstance(parsed, dict):
        logger.error(f"[WEBHOOK] AI reply did not return a dict. Raw reply: {parsed}")
        parsed = {}

    if parsed.get("instruction") in ("close_session_and_notify_sales", "close_session_drop"):
        return process_ai_reply_and_send(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)



    # --- TEMPLATE PROCESSING ---
    if "template" in parsed:
        template_id = parsed["template"]
        template_content = get_template_content(template_id)
        for idx, part in enumerate(template_content):
            content_type = part.get("type")
            content_value = part.get("content")
            if content_type == "text":
                send_wassenger_reply(customer_phone, content_value, device_id, delay_seconds=5)
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", content_value)
            elif content_type == "image":
                caption = part.get("caption") or None
                send_wassenger_reply(
                    customer_phone,
                    content_value,
                    device_id,
                    msg_type="image",
                    caption=caption
                )
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", "[Image sent]")
            elif content_type == "document":
                caption = part.get("caption") or None
                send_wassenger_reply(
                    customer_phone,
                    content_value,
                    device_id,
                    msg_type="media",  # <--- THIS IS THE KEY: msg_type must be 'media' for PDFs/docs
                    caption=caption
                )
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", "[PDF sent]")
            # Optionally: handle other types (video, audio, etc.) here
        
            # Always wait for a delay between template parts
            if idx < len(template_content) - 1:
                time.sleep(5)

    # --- MESSAGE PARTS ---
    msg_lines = []
    if "message" in parsed:
        if isinstance(parsed["message"], list):
            msg_lines = parsed["message"]
        elif isinstance(parsed["message"], str):
            msg_lines = [parsed["message"]]
    elif isinstance(parsed, str):
        msg_lines = [parsed]

    SPLIT_MSG_DELAY = 1  # seconds between each message

    for idx, part in enumerate(msg_lines[:3]):
        if part:
            delay = SPLIT_MSG_DELAY * (idx + 1)  # 7s, 14s, 21s, etc.
            send_wassenger_reply(customer_phone, part, device_id, delay_seconds=delay)
            if bot_id and user and session_id:
                if isinstance(part, dict) and part.get("type") == "image":
                    save_message(bot_id, user, session_id, "out", "[IMAGE]", raw_media_url=part.get("content"))
                else:
                    save_message(bot_id, user, session_id, "out", part)


                    
def find_or_create_customer(phone, name=None):
    customer = Customer.query.filter_by(phone_number=phone).first()
    if not customer:
        customer = Customer(phone_number=phone, name=name)
        db.session.add(customer)
        db.session.commit()
    return customer

def get_or_create_session(customer_id, bot_id):
    session_obj = Session.query.filter_by(customer_id=customer_id, bot_id=bot_id, status='open').first()
    if not session_obj:
        session_obj = Session(
            customer_id=customer_id,
            bot_id=bot_id,
            started_at=datetime.now(),
            status='open',
            context={},
        )
        db.session.add(session_obj)
        db.session.commit()
    return session_obj


def close_session(session, reason, info: dict = None):
    session.ended_at = datetime.now()
    session.status = 'closed'
    if info:
        session.context.update(info)
    session.context['close_reason'] = reason
    db.session.commit()

@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Full incoming message: {json.dumps(data)}")
    logger.info(f"[WEBHOOK] Incoming data: {data}")

    try:
        msg = data["data"]
        msg_type = msg.get("type")
        if msg.get("flow") == "outbound":
            return jsonify({"status": "ignored"}), 200
        # Ignore all group messages (from group chat)
        if (
            "@g.us" in msg.get("from", "")  # sender is a group
            or (msg.get("chat", {}).get("type") == "group")  # chat type is group
            or msg.get("meta", {}).get("isGroup") is True    # meta says is group
        ):
            logger.info(f"[WEBHOOK] Ignored group message from: {msg.get('from')}")
            return jsonify({"status": "ignored_group"}), 200
        

        bot_phone = msg.get("toNumber")
        user_phone = msg.get("fromNumber")
        device_id = data["device"]["id"]

        # 1. Extract bot and message first
        bot = get_bot_by_phone(bot_phone)
        if not bot:
            logger.error(f"[ERROR] No bot found for phone {bot_phone}")
            return jsonify({"error": "Bot not found"}), 404

        msg_type = msg.get("type")
        if msg_type == "audio":
            extract_result, raw_media_url = extract_text_from_message(msg)
            transcript = extract_result["transcript"]
            gpt_reply = extract_result["gpt_reply"]
            # Save the transcript as the 'in' message
            customer = find_or_create_customer(user_phone)
            session = get_or_create_session(customer.id, bot.id)
            session_id = str(session.id)
            save_message(bot.id, user_phone, session_id, "in", transcript, raw_media_url=raw_media_url)
            msg_text = gpt_reply or transcript  # For downstream use (AI, etc)
        else:
            msg_text, raw_media_url = extract_text_from_message(msg)
            customer = find_or_create_customer(user_phone)
            session = get_or_create_session(customer.id, bot.id)
            session_id = str(session.id)
            save_message(bot.id, user_phone, session_id, "in", msg_text, raw_media_url=raw_media_url)


        # 3. Only save incoming message ONCE, after customer/session created
        history = get_latest_history(bot.id, user_phone, session_id)

        # 4. Compose tool and context
        tool_id = decide_tool_with_manager_prompt(bot, history)
        tool = None
        if tool_id and tool_id.lower() != "default":
            for t in get_active_tools_for_bot(bot.id):
                if t.tool_id == tool_id:
                    tool = t
                    break
        logger.info(f"[LOGIC] Tool selected: {tool_id}, tool obj: {tool}")

        context_input = "\n".join([
            f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}"
            for m in history
        ])

        ai_reply = compose_reply(bot, tool, history, context_input)

        # 5. Parse AI reply and handle customer info
        try:
            ai_reply_stripped = strip_json_markdown_blocks(ai_reply)
            parsed = ai_reply if isinstance(ai_reply, dict) else json.loads(ai_reply_stripped)
        except Exception as e:
            logger.error(f"[WEBHOOK] Could not parse AI reply as JSON: {ai_reply} ({e})")
            parsed = {}


        # 6. Update customer info only after parsed
        if "name" in parsed:
            customer.name = parsed["name"]
            db.session.commit()
        # (add more: area, language, etc. if in parsed)

        # --- UNIVERSAL BACKEND CLOSING BLOCK ---
        if parsed.get("instruction") in ("close_session_and_notify_sales", "close_session_drop"):
            # 1. Gather all context to save (ignore reserved fields)
            info_to_save = {}
            for k, v in parsed.items():
                if k not in ("message", "notification", "instruction") and v is not None:
                    info_to_save[k] = v
        
            # 2. Set close reason (won/drop)
            close_reason = parsed.get("close_reason")
            if not close_reason:
                close_reason = "won" if parsed["instruction"] == "close_session_and_notify_sales" else "drop"
        
            # 3. Find and update the active session for this customer + bot
            session_obj = (
                db.session.query(Session)
                .filter_by(bot_id=bot.id, customer_id=customer.id, status="open")
                .order_by(Session.started_at.desc())
                .first()
            )
            if session_obj:
                session_obj.status = "closed"
                session_obj.ended_at = datetime.now()
                session_obj.context = {**session_obj.context, **info_to_save, "close_reason": close_reason}
                db.session.commit()
                logger.info(f"[SESSION] Closed session for user {user_phone}, reason: {close_reason}, info: {info_to_save}")
            else:
                logger.warning(f"[SESSION] Tried to close session, but none found for user {user_phone}, bot {bot.id}")
        
            # 4. Only notify group for won/closed sales
            if parsed["instruction"] == "close_session_and_notify_sales" and "notification" in parsed:
                notify_sales_group(bot, parsed["notification"])
        
            # 5. Always send final message(s) to customer, even if closing
            if "message" in parsed:
                msg_lines = parsed["message"]
                if isinstance(msg_lines, str):
                    msg_lines = [msg_lines]
                for idx, part in enumerate(msg_lines[:3]):
                    if part:
                        delay = 1 * (idx + 1)
                        send_wassenger_reply(user_phone, part, device_id, delay_seconds=delay)
                        save_message(bot.id, user_phone, session_id, "out", part)
        
            return jsonify({"status": "session closed", "reason": close_reason})



        # 8. Only send/process reply if session is NOT closed
        process_ai_reply_and_send(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)
        return jsonify({"status": "ok"})

    except Exception as e:
        logger.error(f"[WEBHOOK] Exception: {e}")
        return jsonify({"error": "Webhook processing error"}), 500

# 9. Main run block at file bottom only!
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
