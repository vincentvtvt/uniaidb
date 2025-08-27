import re
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import requests
import openai
import base64
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_bytes
import cv2
import uuid
import os
from threading import Timer, Thread, Lock
from collections import defaultdict
import anthropic

UTC_PLUS_8 = timezone(timedelta(hours=8))

# === CONFIGURATION ===
# Environment variable to bypass session checks when needed
BYPASS_SESSION_CHECKS = os.getenv("BYPASS_SESSION_CHECKS", "false").lower() == "true"
ENABLE_FOLLOW_UP_RESPONSES = os.getenv("ENABLE_FOLLOW_UP_RESPONSES", "true").lower() == "true"

# === ENHANCED BUFFER MANAGEMENT ===
MESSAGE_BUFFER = defaultdict(list)
TIMER_BUFFER = {}
PROCESSING_FLAGS = {}  # Prevent concurrent processing
BUFFER_START_TIME = {}  # Track buffer age
MESSAGE_HASH_CACHE = {}  # Deduplication
BUFFER_LOCK = Lock()  # Thread safety

# === ERROR TRACKING ===
EXTRACTION_ERRORS = []  # Track extraction failures

# --- Universal JSON Prompt Builder ---
def build_json_prompt(base_prompt, example_json):
    json_instruction = (
        "\n\nAlways respond ONLY with a strict, valid JSON object. "
        "Use double quotes for all keys and string values. "
        "Do NOT include any explanation, markdown, code block formatting, or tags—just pure JSON.\n"
        "Example:\n"
        f"{example_json.strip()}"
    )
    return base_prompt.strip() + json_instruction

# === Time helper for AI context (UTC+8) ===
def get_current_datetime_str_utc8():
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%a, %d %b %Y, %H:%M:%S %Z")
    
def make_timezone_aware(dt):
    """
    Ensure datetime is timezone-aware (UTC+8).
    If naive, assume it's UTC+8 and make it aware.
    If already aware, convert to UTC+8.
    """
    if dt is None:
        return None
    
    if dt.tzinfo is None:
        # Naive datetime - assume UTC+8
        return dt.replace(tzinfo=UTC_PLUS_8)
    else:
        # Already aware - convert to UTC+8
        return dt.astimezone(UTC_PLUS_8)

def get_current_datetime_utc8():
    """
    Get current datetime in UTC+8 (timezone-aware).
    Returns a datetime OBJECT, not a string.
    """
    return datetime.now(UTC_PLUS_8)

def strip_json_markdown_blocks(text):
    """Removes ```json ... ``` or ``` ... ``` wrappers from AI output."""
    return re.sub(r'```[a-z]*\s*([\s\S]*?)```', r'\1', text, flags=re.MULTILINE).strip()

def build_json_prompt_with_reasoning(base_prompt, example_json):
    reasoning_instruction = (
        "Before answering, briefly explain your reasoning for the tool selection in 1-2 sentences. "
        "After your reasoning, output ONLY the strict JSON. Do NOT add code block formatting, markdown, or any tags—just pure JSON.\n"
    )
    json_instruction = (
        reasoning_instruction +
        "Example:\n"
        f"{example_json.strip()}"
    )
    return base_prompt.strip() + "\n\n" + json_instruction

# === MESSAGE DEDUPLICATION ===
def is_duplicate_message(user_phone, msg_text, window_seconds=5):
    """Check if message is duplicate within time window"""
    msg_hash = hashlib.md5(f"{user_phone}:{msg_text}".encode()).hexdigest()
    current_time = time.time()
    
    with BUFFER_LOCK:
        if msg_hash in MESSAGE_HASH_CACHE:
            if current_time - MESSAGE_HASH_CACHE[msg_hash] < window_seconds:
                return True
        
        MESSAGE_HASH_CACHE[msg_hash] = current_time
        # Clean old entries
        keys_to_remove = [k for k, v in MESSAGE_HASH_CACHE.items() 
                         if current_time - v > 60]
        for k in keys_to_remove:
            MESSAGE_HASH_CACHE.pop(k, None)
    
    return False

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("UniAI")

client = anthropic.Anthropic()
openai.api_key = os.getenv("OPENAI_API_KEY")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Models (keeping existing) ---
class Bot(db.Model):
    __tablename__ = 'bots'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    phone_number = db.Column(db.String(30))
    config = db.Column(db.JSON)
    created_at = db.Column(db.DateTime)
    system_prompt = db.Column(db.Text)
    manager_system_prompt = db.Column(db.Text)

class Lead(db.Model):
    __tablename__ = 'leads'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    business_id = db.Column(db.Integer)
    session_id = db.Column(db.String(50))
    name = db.Column(db.String(100), nullable=False)
    contact = db.Column(db.String(50), nullable=False)
    info = db.Column(db.JSON, default={})
    status = db.Column(db.String(20), default='open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    whatsapp_number = db.Column(db.String(50))

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

# === ERROR TRACKING TABLE ===
class ExtractionError(db.Model):
    __tablename__ = 'extraction_errors'
    id = db.Column(db.Integer, primary_key=True)
    message_data = db.Column(db.JSON)
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# === ENHANCED HELPERS ===
def save_extraction_error(msg_data, error_str):
    """Save extraction errors for manual review"""
    try:
        error_record = ExtractionError(
            message_data=msg_data,
            error=error_str,
            created_at=get_current_datetime_utc8()
        )
        db.session.add(error_record)
        db.session.commit()
    except Exception as e:
        logger.error(f"[ERROR TRACKING] Failed to save error: {e}")

def download_file(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def encode_image_b64(img_bytes):
    return base64.b64encode(img_bytes).decode()

def extract_text_from_image(img_url, prompt=None):
    try:
        image_bytes = download_wassenger_media(img_url)
        img_b64 = encode_image_b64(image_bytes)
        logger.info("[VISION] Sending image to OpenAI Vision...")
        messages = [
            {"role": "system", "content": prompt or "Extract all visible text from this image. If no text, describe what you see."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}
        ]
        result = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=8192
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[VISION ERROR] {e}", exc_info=True)
        save_extraction_error({"url": img_url}, str(e))
        return "[Image extraction failed]"

def transcribe_audio_from_url(audio_url):
    try:
        audio_bytes = download_wassenger_media(audio_url)
        if not audio_bytes or len(audio_bytes) < 1024:
            logger.error("[AUDIO DOWNLOAD] Failed or too small")
            return "[audio received, transcription failed]"
        temp_path = "/tmp/temp_audio.ogg"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        with open(temp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        logger.info(f"[WHISPER] Transcript: {transcript.text.strip()}")
        return transcript.text.strip()
    except Exception as e:
        logger.error(f"[WHISPER ERROR] {e}", exc_info=True)
        save_extraction_error({"url": audio_url}, str(e))
        return "[audio received, transcription failed]"

def download_to_bytes(url):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content = resp.content
    logger.info(f"[DOWNLOAD] {len(content)} bytes downloaded from {url}")
    return content

def get_filename_from_url_or_path(input_value, default_ext=".pdf"):
    if isinstance(input_value, str):
        base = os.path.basename(input_value.split("?")[0])
        if "." in base:
            return base
        else:
            return base + default_ext
    else:
        return f"file-{uuid.uuid4().hex}{default_ext}"

# === ENHANCED MESSAGE EXTRACTION WITH ERROR HANDLING ===
def extract_text_from_message(msg):
    """Enhanced extraction with better error handling"""
    import cv2
    import numpy as np
    from pdf2image import convert_from_bytes
    
    extraction_errors = []
    msg_type = msg.get("type")
    media = msg.get("media", {})
    msg_text, media_url = None, None

    def get_media_url(media):
        url = media.get("url")
        if not url and "links" in media and "download" in media["links"]:
            url = "https://api.wassenger.com" + media["links"]["download"]
        return url

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
            extraction_errors.append(str(e))
        return None

    try:
        # Process different message types...
        if msg_type == "sticker":
            img_url = get_media_url(media)
            if img_url:
                try:
                    image_bytes = download_wassenger_media(img_url)
                    if media.get("extension", "").lower() == "webp":
                        im = Image.open(BytesIO(image_bytes)).convert("RGBA")
                        buf = BytesIO()
                        im.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                    img_b64 = encode_image_b64(image_bytes)
                    vision_msg = [
                        {"role": "system", "content": "This is a WhatsApp sticker. Briefly describe what is shown."},
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
                    save_extraction_error(msg, str(e))
                    return "[Sticker received - extraction failed]", img_url
            return "[Sticker received]", None

        elif msg_type == "image":
            img_url = get_media_url(media)
            if img_url:
                try:
                    image_bytes = download_wassenger_media(img_url)
                    img_b64 = encode_image_b64(image_bytes)
                    vision_msg = [
                        {"role": "system", "content": "This is a photo/image received on WhatsApp. Summarize what you see."},
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
                    save_extraction_error(msg, str(e))
                    return "[Image received - extraction failed]", img_url
            return "[Image received, no url]", None

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
                            {"role": "system", "content": "This is the first frame of a WhatsApp video. Summarize the scene."},
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
                    save_extraction_error(msg, str(e))
                    return f"[Video received: {file_name} - extraction failed]", vid_url
            return "[Video received, no url]", None

        elif msg_type == "audio":
            audio_url = get_media_url(media)
            if audio_url:
                try:
                    transcript = transcribe_audio_from_url(audio_url)
                    if transcript and transcript.lower() not in ("[audio received, no url]", "[audio received, transcription failed]"):
                        gpt_prompt = f"This is a WhatsApp audio message transcribed as: '{transcript}'. Reply in a short, natural phrase."
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
                    save_extraction_error(msg, str(e))
                    return {"transcript": "[Audio received - extraction failed]", "gpt_reply": None}, audio_url
            return {"transcript": "[Audio received, no url]", "gpt_reply": None}, None

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
                                {"role": "system", "content": "This is the first page of a PDF document. Summarize what you see."},
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
                            {"role": "system", "content": "This is an image document. Summarize what you see."},
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
                    save_extraction_error(msg, str(e))
                    return f"[Document received: {file_name} - extraction failed]", doc_url
            return "[Document received, no url]", None

        else:
            msg_text = msg.get("body") or msg.get("caption") or f"[{msg_type} received]" if msg_type else "[Message received]"
            return msg_text, None
    
    except Exception as e:
        logger.error(f"[EXTRACTION CRITICAL ERROR] {e}", exc_info=True)
        save_extraction_error(msg, str(e))
        return f"[Message extraction failed: {msg_type}]", None

def get_template_content(template_id):
    template = db.session.query(Template).filter_by(template_id=template_id, active=True).first()
    if not template or not template.content:
        return []
    return template.content if isinstance(template.content, list) else json.loads(template.content)

def download_wassenger_media(url):
    headers = {"Token": os.getenv("WASSENGER_API_KEY")}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.error(f"[WASSENGER MEDIA DOWNLOAD ERROR] {e}")
        return None

def save_lead(name, contact, info_dict, whatsapp_number, bot_id=None, business_id=None, session_id=None, status="open"):
    lead = Lead(
        name=name,
        contact=contact,
        whatsapp_number=whatsapp_number,
        info=info_dict,
        bot_id=bot_id,
        business_id=business_id,
        session_id=session_id,
        status=status
    )
    db.session.add(lead)
    db.session.commit()
    return lead

def upload_and_send_media(recipient, file_url_or_path, device_id, caption=None, msg_type=None, delay_seconds=5):
    filename = None
    if not msg_type:
        if isinstance(file_url_or_path, str):
            ext = os.path.splitext(file_url_or_path.split("?")[0])[1].lower()
            if ext in ('.jpg', '.jpeg', '.png', '.webp'):
                msg_type = "image"
            elif ext in ('.pdf', '.doc', '.docx'):
                msg_type = "media"
            else:
                msg_type = "media"
            filename = os.path.basename(file_url_or_path.split("?")[0])
        else:
            msg_type = "media"
            filename = "file"
    else:
        if isinstance(file_url_or_path, str) and not filename:
            filename = os.path.basename(file_url_or_path.split("?")[0])

    if isinstance(file_url_or_path, str) and len(file_url_or_path) == 24 and file_url_or_path.isalnum():
        file_id = file_url_or_path
    else:
        file_id = upload_any_file_to_wassenger(file_url_or_path, filename=filename)

    if not file_id:
        logger.error(f"[UPLOAD & SEND] Failed to upload file for recipient {recipient}")
        return None

    return send_wassenger_reply(
        recipient,
        file_id,
        device_id,
        msg_type=msg_type,
        caption=caption,
        delay_seconds=delay_seconds
    )

def upload_any_file_to_wassenger(file_path_or_bytes, filename=None, msg_type=None):
    url = "https://api.wassenger.com/v1/files"
    headers = {"Token": WASSENGER_API_KEY}

    if isinstance(file_path_or_bytes, str) and not file_path_or_bytes.startswith("http"):
        if not filename:
            filename = os.path.basename(file_path_or_bytes)
        with open(file_path_or_bytes, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_path_or_bytes, str) and file_path_or_bytes.startswith("http"):
        file_bytes = download_to_bytes(file_path_or_bytes)
        if not filename:
            filename = get_filename_from_url_or_path(file_path_or_bytes, default_ext=".pdf" if msg_type == "media" else ".jpg")
    else:
        file_bytes = file_path_or_bytes
        if not filename:
            ext = ".pdf" if msg_type == "media" else ".jpg"
            filename = f"file-{uuid.uuid4().hex}{ext}"

    logger.debug(f"[UPLOAD DEBUG] filename: {filename}, size: {len(file_bytes)}")
    
    files = {"file": (filename, file_bytes)}

    try:
        resp = requests.post(url, headers=headers, files=files, timeout=60)
        if resp.status_code == 409:
            logger.warning(f"[MEDIA UPLOAD] Duplicate file detected (409 Conflict).")
            return None
        resp.raise_for_status()
        resp_json = resp.json()
        if isinstance(resp_json, list) and resp_json and resp_json[0].get('id'):
            file_id = resp_json[0]['id']
        elif isinstance(resp_json, dict) and resp_json.get('id'):
            file_id = resp_json['id']
        else:
            logger.error(f"[MEDIA UPLOAD FAIL] Bad response: {resp.text}")
            return None
        logger.info(f"[MEDIA UPLOAD SUCCESS] file_id: {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"[MEDIA UPLOAD FAIL] Error: {e}")
        return None

def send_wassenger_reply(phone, text, device_id, delay_seconds=0, msg_type="text", caption=None, deliver_at_iso=None):
    scheduled_time = deliver_at_iso or (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    url = "https://api.wassenger.com/v1/messages"
    headers = {"Content-Type": "application/json", "Token": WASSENGER_API_KEY}
    payload = {"device": device_id}

    if isinstance(phone, str) and phone.endswith("@g.us"):
        payload["group"] = phone
    else:
        payload["phone"] = phone

    if msg_type == "text":
        payload["message"] = text
        payload["deliverAt"] = scheduled_time
        payload["order"] = True
    elif msg_type in ("image", "media"):
        if isinstance(text, str) and len(text) == 24 and text.isalnum():
            payload["media"] = {"file": text}
        else:
            if isinstance(text, str) and text.startswith("http"):
                file_bytes = download_to_bytes(text)
                filename = "image.jpg" if msg_type == "image" else "document.pdf"
                file_id = upload_any_file_to_wassenger(file_bytes, filename=filename)
            else:
                file_id = upload_any_file_to_wassenger(text)
            if not file_id:
                logger.error(f"[SEND {msg_type.upper()}] Failed to upload")
                return None
            payload["media"] = {"file": file_id}
        if caption:
            payload["message"] = caption
        payload["deliverAt"] = scheduled_time
        payload["order"] = True
    else:
        logger.error(f"Unsupported msg_type: {msg_type}")
        return None

    payload = {k: v for k, v in payload.items() if v is not None}
    logger.debug(f"[WASSENGER PAYLOAD]: {payload}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        logger.info(f"Wassenger response: {resp.status_code} {resp.text}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SEND WASSENGER ERROR] {e}")
        return None

# === ENHANCED MESSAGE SENDING WITH SESSION CHECK ===
def send_messages_with_anchor(phone, lines, device_id, bot_id=None, session_id=None, gap_seconds=3):
    """Enhanced version with configurable session status checking"""
    if not lines:
        return
    
    lines = [l for l in (lines if isinstance(lines, list) else [str(lines)]) if l]
    
    # Check session status before sending (can be bypassed)
    if session_id and not BYPASS_SESSION_CHECKS:
        session = db.session.query(Session).filter_by(id=int(session_id)).first()
        if session:
            db.session.refresh(session)  # Force reload from DB
            if session.status == 'closed':
                logger.info(f"[ANCHOR SEND] Session {session_id} is closed. Skipping messages.")
                # If bypassed, log but continue
                if BYPASS_SESSION_CHECKS:
                    logger.info(f"[ANCHOR SEND] BYPASS_SESSION_CHECKS is enabled, sending anyway")
                else:
                    return
    
    # Send first message
    first_text = lines[0]
    first_resp = send_wassenger_reply(phone, first_text, device_id, delay_seconds=1, msg_type="text")
    
    try:
        created_at = first_resp.get("createdAt") if isinstance(first_resp, dict) else None
        base_time = datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else datetime.now(timezone.utc)
    except Exception:
        base_time = datetime.now(timezone.utc)
    
    if bot_id and session_id:
        save_message_safe(bot_id, phone, session_id, "out", first_text)
    
    # Send subsequent messages with session check
    for idx, part in enumerate(lines[1:], start=1):
        # Re-check session status before each message (can be bypassed)
        if session_id and not BYPASS_SESSION_CHECKS:
            session = db.session.query(Session).filter_by(id=int(session_id)).first()
            if session:
                db.session.refresh(session)
                if session.status == 'closed':
                    logger.info(f"[ANCHOR SEND] Session closed during sending. Stopping at message {idx+1}")
                    if BYPASS_SESSION_CHECKS:
                        logger.info(f"[ANCHOR SEND] BYPASS_SESSION_CHECKS is enabled, continuing")
                    else:
                        break
        
        try:
            deliver_at = (base_time + timedelta(seconds=idx * gap_seconds)).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
            send_wassenger_reply(phone, part, device_id, msg_type="text", deliver_at_iso=deliver_at)
            if bot_id and session_id:
                save_message_safe(bot_id, phone, session_id, "out", part)
        except Exception as e:
            logger.error(f"[ANCHOR SEND ERROR] idx={idx} err={e}")

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
        logger.warning("[NOTIFY] Notification group or device_id missing")

def get_bot_by_phone(phone_number):
    num_variants = [
        phone_number,
        phone_number.lstrip('+'),
        phone_number.replace('+', '').replace('@c.us', ''),
        '+' + phone_number.replace('@c.us', '').lstrip('+'),
        phone_number.replace('@c.us', ''),
    ]
    logger.debug(f"[DB] Bot lookup for: {phone_number}")
    for variant in num_variants:
        bot = Bot.query.filter_by(phone_number=variant).first()
        if bot:
            logger.debug(f"[DB] Bot found! Matched: {variant}")
            return bot
    logger.error(f"[DB] Bot NOT FOUND for: {phone_number}")
    return None

def get_active_tools_for_bot(bot_id):
    tools = (
        db.session.query(Tool)
        .join(BotTool, (Tool.tool_id == BotTool.tool_id) & (BotTool.bot_id == bot_id) & (Tool.active == True) & (BotTool.active == True))
        .all()
    )
    logger.info(f"[DB] Tools for bot_id={bot_id}: {[t.tool_id for t in tools]}")
    return tools

# === SAFE MESSAGE SAVING WITH RETRY ===
def save_message_safe(bot_id, customer_phone, session_id, direction, content, raw_media_url=None):
    """Save message with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            msg = Message(
                bot_id=bot_id,
                customer_phone=customer_phone,
                session_id=session_id,
                direction=direction,
                content=content,
                raw_media_url=raw_media_url,
                created_at=get_current_datetime_utc8()
            )
            db.session.add(msg)
            db.session.commit()
            logger.info(f"[DB] Saved message ({direction})")
            return True
        except Exception as e:
            logger.error(f"[DB] Save message attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                db.session.rollback()
                time.sleep(0.5)
            else:
                # Save to backup file
                with open('failed_messages.log', 'a') as f:
                    f.write(f"{datetime.now()},{bot_id},{customer_phone},{direction},{content}\n")
                return False

def save_message(bot_id, customer_phone, session_id, direction, content, raw_media_url=None):
    """Original save_message for compatibility"""
    return save_message_safe(bot_id, customer_phone, session_id, direction, content, raw_media_url)

def get_latest_history(bot_id, customer_phone, session_id, n=100):
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
    history_text = "Current date/time (UTC+8): " + get_current_datetime_str_utc8() + "\n" + "\n".join(
        [f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}" for m in history]
    )

    tool_menu = build_tool_menu_for_prompt(bot.id)
    tool_menu_text = (
        "Here are the available tools you can select (ID, name, and description):\n"
        f"{tool_menu}\n"
        "Choose the single most appropriate tool for this conversation."
    )

    manager_prompt = build_json_prompt_with_reasoning(
        (bot.manager_system_prompt or "") + "\n" + tool_menu_text,
        '{\n  "TOOLS": "Default"\n}',
    )
    
    logger.info(f"[AI DECISION] Starting tool selection")
    
    try:
        response = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.3,
            system=manager_prompt,
            messages=[{"role": "user", "content": history_text}]
        )
        
        raw_response = response.content[0].text
        logger.info(f"[AI DECISION] Raw response: {raw_response}")
        
        reasoning_match = re.search(r'(.*?)(?=\{)', raw_response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            logger.info(f"[AI TOOL DECISION REASONING]: {reasoning}")
        
        json_block = strip_json_markdown_blocks(raw_response)
        json_match = re.search(r'\{[^}]*"TOOLS"[^}]*\}', json_block)
        if json_match:
            json_str = json_match.group(0)
            tool_decision = json.loads(json_str)
            tools_value = tool_decision.get("TOOLS")
            logger.info(f"[AI DECISION] Selected tool: {tools_value}")
            return tools_value
        else:
            logger.error(f"[AI DECISION] No valid JSON found")
            return "Default"
            
    except json.JSONDecodeError as e:
        logger.error(f"[AI DECISION] JSON decode error: {e}")
        return "Default"
    except Exception as e:
        logger.error(f"[AI DECISION] Unexpected error: {e}")
        return "Default"

def compose_reply(bot, tool, history, context_input):
    if tool:
        prompt = (bot.system_prompt or "") + "\n" + (tool.prompt or "")
        example_json = '''{
  "template": "bf_UGnkL24bhtCQBIJr7hbT",
  "message": [
    "example template 1",
    "example template 2",
    "example template 3"
  ]
}'''
    else:
        prompt = bot.system_prompt or ""
        example_json = '''{
  "message": [
    "example message 1",
    "example message 2",
    "example message 3"
  ]
}'''
    
    reply_prompt = build_json_prompt(prompt, example_json)
    logger.info(f"[AI REPLY] Starting composition")
    
    try:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.7,
            system=reply_prompt,
            messages=[{"role": "user", "content": context_input}]
        ) as stream:
            reply_accum = ""
            print("[STREAM] Streaming model reply:")
            for text in stream.text_stream:
                reply_accum += text
                print(text, end="", flush=True)
            
            logger.info(f"\n[AI REPLY STREAMED]: {reply_accum}")
            cleaned_response = strip_json_markdown_blocks(reply_accum)
            
            try:
                tool_decision = json.loads(cleaned_response)
                return tool_decision
            except json.JSONDecodeError as e:
                logger.error(f"[AI REPLY] JSON decode error: {e}")
                return {"message": ["I apologize, there was an error processing your request. Please try again."]}
                
    except Exception as e:
        logger.error(f"[AI REPLY] Unexpected error: {e}")
        return {"message": ["I apologize, there was an error processing your request. Please try again."]}

# === ENHANCED AI REPLY PROCESSING WITH SESSION CHECKS ===
def is_session_still_open(session_id):
    """Helper to check if session is still open"""
    if not session_id:
        return True
    if BYPASS_SESSION_CHECKS:
        return True  # Always return True if bypass is enabled
    session = db.session.query(Session).filter_by(id=int(session_id)).first()
    if session:
        db.session.refresh(session)  # Force reload from DB
        return session.status == 'open'
    return False

def process_ai_reply_and_send(customer_phone, ai_reply, device_id, bot_id=None, user=None, session_id=None):
    """Enhanced with multiple session status checks"""
    
    # Initial session check
    if not is_session_still_open(session_id):
        logger.info(f"[AI REPLY] Session {session_id} is already closed. Skipping.")
        return

    def extract_field_from_notification(notification, field):
        if not notification:
            return None
        pattern = r'{field}[:：]\s*([^\n]+)'.format(field=re.escape(field))
        m = re.search(pattern, notification)
        return m.group(1).strip() if m else None

    try:
        parsed = ai_reply if isinstance(ai_reply, dict) else json.loads(ai_reply)
    except Exception as e:
        logger.error(f"[WEBHOOK] Could not parse AI reply: {e}")
        parsed = {}

    if not isinstance(parsed, dict):
        logger.error(f"[WEBHOOK] AI reply not a dict")
        parsed = {}

    # Handle session closing instructions
    if parsed.get("instruction") in ("close_session_and_notify_sales", "close_session_drop"):
        logger.info("[AI REPLY] Session close detected")
        
        # Re-check session status before processing close
        if not is_session_still_open(session_id):
            logger.info(f"[AI REPLY] Session already closed, skipping close logic")
            return
        
        info_to_save = {}
        for k, v in parsed.items():
            if k not in ("message", "notification", "instruction") and v is not None:
                info_to_save[k] = v
        
        close_reason = parsed.get("close_reason")
        if not close_reason:
            close_reason = "won" if parsed["instruction"] == "close_session_and_notify_sales" else "drop"
        
        lose_reason = (
            parsed.get("lose_reason") or 
            parsed.get("drop_reason") or 
            info_to_save.get("lose_reason") or 
            info_to_save.get("drop_reason")
        )
        
        if close_reason in ("drop", "lost", "lose"):
            if lose_reason:
                close_reason = f"{close_reason}: {lose_reason}"
                info_to_save["loss_analysis"] = lose_reason
                logger.info(f"[LOSS ANALYSIS] Reason: {lose_reason}")
            else:
                close_reason = f"{close_reason}: not specified"
        
        # Find and update session
        bot = db.session.get(Bot, bot_id) if bot_id else None
        customer = Customer.query.filter_by(phone_number=customer_phone).first()
        
        session_obj = None
        if bot and customer:
            session_obj = (
                db.session.query(Session)
                .filter_by(bot_id=bot.id, customer_id=customer.id, status="open")
                .order_by(Session.started_at.desc())
                .first()
            )
        
        # Update session with lock to prevent race conditions
        if session_obj:
            with BUFFER_LOCK:
                # Final check before closing
                db.session.refresh(session_obj)
                if session_obj.status == "closed":
                    logger.info(f"[SESSION CLOSE] Session already closed by another process")
                    return
                if not session_obj.context:
                    session_obj.context = {}
                session_obj.context["close_reason"] = close_reason
                session_obj.context.update(info_to_save)
                db.session.commit()
                logger.info(f"[SESSION CLOSE] Session {session_obj.id} closed")
        
        # Handle lead creation and notification
        notification = parsed.get("notification") or info_to_save.get("notification") or ""
        name = parsed.get("name") or info_to_save.get("name")
        if not name:
            name_match = re.search(r'Customer name[:：]\s*([^\n]+)', notification)
            if name_match:
                name = name_match.group(1).strip()
        
        contact = (
            parsed.get("contact") or 
            info_to_save.get("contact") or 
            extract_field_from_notification(notification, "Contact") or 
            extract_field_from_notification(notification, "Phone")
        )
        if not contact or str(contact).strip().lower() in [
            "whatsapp number", "[whatsapp number]", "same", "this", "use this", "ok", "yes"
        ]:
            contact = customer_phone
        
        info_fields = {}
        for k, v in {**parsed, **info_to_save}.items():
            if k not in ["name", "contact"] and v is not None:
                info_fields[k] = v
        
        if (
            parsed.get("instruction") == "close_session_and_notify_sales" and 
            close_reason.startswith("won") and 
            name and contact
        ):
            lead = Lead(
                name=name,
                contact=contact,
                whatsapp_number=customer_phone,
                info=info_fields,
                bot_id=bot_id,
                business_id=getattr(bot, 'business_id', None),
                session_id=session_obj.id if session_obj else None,
                status="open"
            )
            db.session.add(lead)
            db.session.commit()
            logger.info(f"[LEAD] Lead saved: {lead.id}")
            
            # Notify sales
            cfg = bot.config
            if isinstance(cfg, str):
                import json as _json
                cfg = _json.loads(cfg) if cfg else {}
            notify_whatsapp = (cfg or {}).get("notification_group") or ""
            
            header = f"{customer_phone}\n\n{notify_whatsapp}".strip()
            notification_text = parsed.get("notification") or ""
            
            whitelist = ["desired_service", "industry", "area", "facebook_link", "tiktok_link", "current_challenges"]
            control_keys = {"message", "notification", "instruction", "template"}
            
            summary_lines = [f"{k}: {info_fields[k]}" for k in whitelist if info_fields.get(k)]
            other_lines = [
                f"{k}: {v}" for k, v in info_fields.items()
                if k not in set(whitelist) | control_keys and v not in (None, "", [])
            ]
            
            parts = [header]
            if notification_text:
                parts.append(notification_text)
            if summary_lines or other_lines:
                parts.append("\n".join(summary_lines + other_lines))
            final_note = "\n\n".join(parts)
            
            logger.info(f"[NOTIFY SALES]: {final_note}")
            notify_sales_group(bot, final_note)
        
        # Send customer messages
        if "message" in parsed:
            send_messages_with_anchor(customer_phone, parsed["message"], device_id, bot_id=bot_id, session_id=session_id, gap_seconds=3)
        
        return
    
    # Normal message sending
    if "message" in parsed and isinstance(parsed["message"], list):
        send_messages_with_anchor(customer_phone, parsed["message"], device_id, bot_id=bot_id, session_id=session_id, gap_seconds=3)
    elif "message" in parsed:
        send_wassenger_reply(customer_phone, str(ai_reply), device_id, delay_seconds=5, msg_type="text")
        if bot_id and user and session_id:
            save_message_safe(bot_id, customer_phone, session_id, "out", str(ai_reply))
    
    # Template processing
    if "template" in parsed:
        template_id = parsed["template"]
        template_content = get_template_content(template_id)
        doc_counter = 1
        img_counter = 1
        for idx, part in enumerate(template_content):
            # Check session before each template part
            if not is_session_still_open(session_id):
                logger.info(f"[TEMPLATE] Session closed during template sending")
                break
                
            content_type = part.get("type")
            content_value = part.get("content")
            caption = part.get("caption") or None
            delay = max(0, idx * 25)
            
            if content_type == "text":
                send_wassenger_reply(customer_phone, content_value, device_id, delay_seconds=delay)
                if bot_id and user and session_id:
                    save_message_safe(bot_id, user, session_id, "out", content_value)
            elif content_type == "image":
                filename = f"image{img_counter}.jpg"
                img_counter += 1
                file_id = upload_any_file_to_wassenger(content_value, filename=filename)
                if file_id:
                    send_wassenger_reply(customer_phone, file_id, device_id, msg_type="image", caption=caption, delay_seconds=delay)
                    if bot_id and user and session_id:
                        save_message_safe(bot_id, user, session_id, "out", "[Image sent]")
            elif content_type == "document":
                filename = f"document{doc_counter}.pdf"
                doc_counter += 1
                file_id = upload_any_file_to_wassenger(content_value, filename=filename)
                if file_id:
                    send_wassenger_reply(customer_phone, file_id, device_id, msg_type="media", caption=caption, delay_seconds=delay)
                    if bot_id and user and session_id:
                        save_message_safe(bot_id, user, session_id, "out", "[PDF sent]")

                session_obj.status = "closed"
                session_obj.ended_at = get_current_datetime_utc8()

def find_or_create_customer(phone, name=None):
    customer = Customer.query.filter_by(phone_number=phone).first()
    if not customer:
        customer = Customer(phone_number=phone, name=name)
        db.session.add(customer)
        db.session.commit()
    return customer

def check_recent_closed_session(customer_id, bot_id, days_threshold=14):
    current_time = get_current_datetime_utc8()
    cutoff_date = current_time - timedelta(days=days_threshold)
    
    recent_session = (
        Session.query
        .filter_by(customer_id=customer_id, bot_id=bot_id, status='closed')
        .order_by(Session.ended_at.desc())
        .first()
    )
    
    if recent_session and recent_session.ended_at:
        ended_at_aware = make_timezone_aware(recent_session.ended_at)
        if ended_at_aware >= cutoff_date:
            return recent_session
    
    return None

def get_or_create_session(customer_id, bot_id):
    session_obj = Session.query.filter_by(customer_id=customer_id, bot_id=bot_id, status='open').first()
    
    if not session_obj:
        recent_closed = check_recent_closed_session(customer_id, bot_id, days_threshold=14)
        
        if recent_closed:
            recent_closed.is_recently_closed = True
            return recent_closed
        
        session_obj = Session(
            customer_id=customer_id,
            bot_id=bot_id,
            started_at=get_current_datetime_utc8(),
            status='open',
            context={},
        )
        db.session.add(session_obj)
        db.session.commit()
    
    if session_obj.status == 'closed':
        session_obj.is_already_closed = True
    
    return session_obj

def generate_processing_message(customer_language=None):
    messages = {
        'en': "Thank you for reaching out. We are currently processing your previous request. Our team will contact you soon with an update.",
        'ms': "Terima kasih kerana menghubungi kami. Kami sedang memproses permintaan anda yang sebelumnya. Pasukan kami akan menghubungi anda tidak lama lagi dengan kemaskini.",
        'zh': "感谢您的联系。我们目前正在处理您之前的请求。我们的团队将很快与您联系并提供更新。",
        'ta': "தொடர்பு கொண்டதற்கு நன்றி. உங்கள் முந்தைய கோரிக்கையை நாங்கள் தற்போது செயலாக்கி வருகிறோம்.",
        'default': "Thank you for reaching out. We are currently processing your previous request. Our team will contact you soon with an update."
    }
    
    language = customer_language or 'default'
    return messages.get(language, messages['default'])

def close_session(session, reason, info: dict = None):
    session.ended_at = get_current_datetime_utc8()
    session.status = 'closed'
    if info:
        session.context.update(info)
    session.context['close_reason'] = reason
    db.session.commit()

def detect_customer_intent(message_text, session_context, bot):
    """
    Enhanced intent detection with better follow-up handling
    Returns: 'new_request', 'follow_up', or 'unclear'
    """
    
    # Keywords that indicate NEW request
    new_request_keywords = [
        'another', 'different', 'new', 'other', 'else', 'lagi', 'baru', 'lain',
        'change', 'switch', 'upgrade', 'downgrade', 'cancel and', 'instead',
        'also want', 'additionally', 'one more', 'extra', 'tambah', 'tukar'
    ]
    
    # Keywords that indicate FOLLOW-UP
    follow_up_keywords = [
        'status', 'update', 'how about', 'already', 'when', 'bila', 'follow',
        'check', 'confirm', 'still waiting', 'pending', 'progress', 'any news',
        'heard back', 'reply', 'response', 'answered', 'contacted', 'called'
    ]
    
    # Use AI for more sophisticated detection
    try:
        # Get previous service/product from session context
        previous_service = session_context.get('desired_service', 'unknown service')
        close_reason = session_context.get('close_reason', 'unknown')
        
        # Build prompt for AI intent detection
        intent_prompt = f"""
        Analyze this customer message to determine their intent.
        
        Previous context:
        - Service discussed: {previous_service}
        - Session close reason: {close_reason}
        - Days since closed: Within 14 days
        
        Customer's new message: "{message_text}"
        
        Determine if this is:
        1. "new_request" - Customer wants a DIFFERENT or ADDITIONAL service/product
        2. "follow_up" - Customer is asking about their EXISTING request/application
        3. "unclear" - Cannot determine intent clearly
        
        Look for:
        - New request indicators: asking about different products, new requirements, additional services
        - Follow-up indicators: asking for status, updates, when they'll be contacted, checking on progress
        
        Respond with ONLY one word: new_request, follow_up, or unclear
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an intent classifier. Reply with only: new_request, follow_up, or unclear"},
                {"role": "user", "content": intent_prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        ai_intent = response.choices[0].message.content.strip().lower()
        logger.info(f"[INTENT DETECTION] AI detected: {ai_intent} for message: {message_text[:50]}...")
        
        # Validate AI response
        if ai_intent in ['new_request', 'follow_up', 'unclear']:
            return ai_intent
        
    except Exception as e:
        logger.error(f"[INTENT DETECTION] AI error: {e}")
    
    # Fallback to keyword detection if AI fails
    message_lower = message_text.lower()
    
    # Check for explicit new request keywords
    new_score = sum(1 for keyword in new_request_keywords if keyword in message_lower)
    follow_score = sum(1 for keyword in follow_up_keywords if keyword in message_lower)
    
    logger.info(f"[INTENT DETECTION] Keyword scores - New: {new_score}, Follow-up: {follow_score}")
    
    if new_score > follow_score:
        return 'new_request'
    elif follow_score > new_score:
        return 'follow_up'
    else:
        return 'unclear'

# === ENHANCED BUFFER PROCESSING WITH CONCURRENCY CONTROL ===
def process_buffered_messages(buffer_key):
    """Enhanced with processing flags and better error handling"""
    with app.app_context():
        # Check if already processing
        with BUFFER_LOCK:
            if PROCESSING_FLAGS.get(buffer_key, False):
                logger.info(f"[BUFFER] Already processing {buffer_key}, skipping")
                return
            PROCESSING_FLAGS[buffer_key] = True
        
        try:
            bot_id, user_phone, session_id = buffer_key
            bot = db.session.get(Bot, bot_id)
            
            # Get messages from buffer
            messages = MESSAGE_BUFFER.pop(buffer_key, [])
            if not messages:
                logger.info(f"[BUFFER] No messages for {buffer_key}")
                return
            
            device_id = messages[-1].get("device_id")
            
            # Check session status
            session = db.session.query(Session).filter_by(id=int(session_id)).first()
            if session:
                db.session.refresh(session)
                if session.status == 'closed' and not BYPASS_SESSION_CHECKS:
                    logger.info(f"[BUFFER PROCESS] Session {session_id} is closed. Skipping AI.")
                    return
            
            # Combine messages
            combined_text = "\n".join(m['msg_text'] for m in messages if m['msg_text'])
            history = get_latest_history(bot_id, user_phone, session_id)
            
            session_status_note = ""
            if session and session.status == 'closed':
                session_status_note = "\n[SYSTEM NOTE: This session is already closed. Do not generate close_session instructions.]"
            
            context_input = (
                "Current date/time (UTC+8): " + get_current_datetime_str_utc8() + 
                session_status_note + "\n" + 
                "\n".join([
                    f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}"
                    for m in history
                ] + [f"User: {combined_text}"])
            )
            
            # Get tool and compose reply
            tool_id = decide_tool_with_manager_prompt(bot, history)
            tool = None
            if tool_id and tool_id.lower() != "default":
                for t in get_active_tools_for_bot(bot.id):
                    if t.tool_id == tool_id:
                        tool = t
                        break
            
            ai_reply = compose_reply(bot, tool, history, context_input)
            process_ai_reply_and_send(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)
            
        except Exception as e:
            logger.error(f"[BUFFER PROCESS ERROR] {e}", exc_info=True)
        finally:
            # Clean up
            with BUFFER_LOCK:
                PROCESSING_FLAGS[buffer_key] = False
                TIMER_BUFFER.pop(buffer_key, None)
                BUFFER_START_TIME.pop(buffer_key, None)

# === ASYNC WEBHOOK PROCESSING ===
def process_webhook_async(data):
    """Enhanced webhook processing with better intent detection and session management"""
    with app.app_context():
        try:
            msg = data["data"]
            msg_type = msg.get("type")
            
            if msg.get("flow") == "outbound":
                return
                
            # Ignore group messages
            if (
                "@g.us" in msg.get("from", "") or 
                (msg.get("chat", {}).get("type") == "group") or 
                msg.get("meta", {}).get("isGroup") is True
            ):
                logger.info(f"[WEBHOOK] Ignored group message")
                return
            
            bot_phone = msg.get("toNumber")
            user_phone = msg.get("fromNumber")
            device_id = data["device"]["id"]
            
            # Get bot
            bot = get_bot_by_phone(bot_phone)
            if not bot:
                logger.error(f"[ERROR] No bot found for {bot_phone}")
                return
            
            # Extract message content
            if msg_type == "audio":
                extract_result, raw_media_url = extract_text_from_message(msg)
                transcript = extract_result["transcript"]
                gpt_reply = extract_result["gpt_reply"]
                msg_text = gpt_reply or transcript
            else:
                msg_text, raw_media_url = extract_text_from_message(msg)
            
            # === Handle special triggers FIRST ===
            if msg_text and msg_text.strip() == "*.*":
                logger.info(f"[TRIGGER] Clear conversation triggered for {user_phone}")
                Message.query.filter_by(bot_id=bot.id, customer_phone=user_phone).delete()
                
                customer = Customer.query.filter_by(phone_number=user_phone).first()
                if customer:
                    existing_session = Session.query.filter_by(
                        customer_id=customer.id, 
                        bot_id=bot.id, 
                        status='open'
                    ).first()
                    if existing_session:
                        existing_session.status = "closed"
                        existing_session.ended_at = get_current_datetime_utc8() - timedelta(days=15)
                        db.session.commit()
                
                customer = find_or_create_customer(user_phone)
                new_session = Session(
                    customer_id=customer.id,
                    bot_id=bot.id,
                    started_at=get_current_datetime_utc8(),
                    status='open',
                    context={},
                )
                db.session.add(new_session)
                db.session.commit()
                
                send_wassenger_reply(user_phone, "Conversation cleared. Starting fresh!", device_id, delay_seconds=1)
                return
            
            # NEW: Force open trigger
            if msg_text and msg_text.strip().lower() == "*open*":
                logger.info(f"[TRIGGER] Force open triggered for {user_phone}")
                customer = find_or_create_customer(user_phone)
                
                # Close any existing session
                existing = Session.query.filter_by(
                    customer_id=customer.id,
                    bot_id=bot.id,
                    status='open'
                ).first()
                if existing:
                    existing.status = "closed"
                    existing.ended_at = get_current_datetime_utc8()
                    db.session.commit()
                
                # Create new open session
                new_session = Session(
                    customer_id=customer.id,
                    bot_id=bot.id,
                    started_at=get_current_datetime_utc8(),
                    status='open',
                    context={'manually_reopened': True},
                )
                db.session.add(new_session)
                db.session.commit()
                
                send_wassenger_reply(user_phone, "Session reopened. Ready to assist!", device_id, delay_seconds=1)
                return
            
            if msg_text and msg_text.strip().lower() == "*off*":
                logger.info(f"[TRIGGER] Force close triggered for {user_phone}")
                customer = find_or_create_customer(user_phone)
                session = Session.query.filter_by(
                    customer_id=customer.id,
                    bot_id=bot.id,
                    status='open'
                ).first()
                if session:
                    session.status = "closed"
                    session.ended_at = get_current_datetime_utc8()
                    session.context['close_reason'] = "force_closed"
                    db.session.commit()
                    note = f"Session for {user_phone} was force closed by agent/trigger."
                    notify_sales_group(bot, note)
                send_wassenger_reply(user_phone, "Session closed by agent.", device_id, delay_seconds=1)
                return
            
            # === Check for duplicates ===
            msg_unique_key = f"{user_phone}:{msg_text}:{msg.get('id', '')}:{msg.get('timestamp', '')}"
            msg_hash = hashlib.md5(msg_unique_key.encode()).hexdigest()
            
            current_time = time.time()
            with BUFFER_LOCK:
                if msg_hash in MESSAGE_HASH_CACHE:
                    last_processed = MESSAGE_HASH_CACHE[msg_hash]
                    if current_time - last_processed < 10:
                        logger.info(f"[WEBHOOK] Duplicate message ignored")
                        return
                MESSAGE_HASH_CACHE[msg_hash] = current_time
                
                # Clean old entries
                keys_to_remove = [k for k, v in MESSAGE_HASH_CACHE.items() 
                                 if current_time - v > 60]
                for k in keys_to_remove:
                    MESSAGE_HASH_CACHE.pop(k, None)
            
            # Get customer and session
            customer = find_or_create_customer(user_phone)
            session = get_or_create_session(customer.id, bot.id)
            
            # Handle already closed sessions
            if hasattr(session, 'is_already_closed') and session.is_already_closed:
                logger.info(f"[SESSION] Already closed for {user_phone}")
                return
            
            # === ENHANCED: Handle recently closed sessions with BETTER INTENT DETECTION ===
            if hasattr(session, 'is_recently_closed') and session.is_recently_closed:
                current_time_dt = get_current_datetime_utc8()
                ended_at = make_timezone_aware(session.ended_at)
                
                if ended_at:
                    days_since_closed = (current_time_dt - ended_at).days
                else:
                    days_since_closed = 0
                
                logger.info(f"[SESSION] Recently closed session ({days_since_closed} days ago)")
                
                # === ENHANCED: Detect customer intent ===
                intent = detect_customer_intent(msg_text, session.context or {}, bot)
                logger.info(f"[INTENT] Customer intent detected: {intent}")
                
                if intent == 'new_request':
                    # Customer wants NEW service - open fresh session
                    logger.info(f"[SESSION] Customer wants NEW service, opening fresh session")
                    
                    # Close the old session properly (mark it as superseded)
                    session.status = "closed"
                    session.ended_at = get_current_datetime_utc8() - timedelta(days=15)  # Bypass 14-day check
                    if not session.context:
                        session.context = {}
                    session.context['superseded_by_new_request'] = True
                    db.session.commit()
                    
                    # Create new session
                    new_session = Session(
                        customer_id=customer.id,
                        bot_id=bot.id,
                        started_at=get_current_datetime_utc8(),
                        status='open',
                        context={'previous_session_id': session.id},
                    )
                    db.session.add(new_session)
                    db.session.commit()
                    
                    session_id = str(new_session.id)
                    
                    # Save the message and process normally
                    save_message_safe(bot.id, user_phone, session_id, "in", msg_text, raw_media_url)
                    
                    # Add to buffer for AI processing
                    buffer_key = (bot.id, user_phone, session_id)
                    current_timestamp = time.time()
                    
                    with BUFFER_LOCK:
                        BUFFER_START_TIME[buffer_key] = current_timestamp
                        MESSAGE_BUFFER[buffer_key] = [{
                            "msg_text": msg_text,
                            "raw_media_url": raw_media_url,
                            "created_at": get_current_datetime_utc8().isoformat(),
                            "device_id": device_id,
                        }]
                        
                        # Process after buffer time
                        if buffer_key in TIMER_BUFFER and TIMER_BUFFER[buffer_key]:
                            TIMER_BUFFER[buffer_key].cancel()
                        TIMER_BUFFER[buffer_key] = Timer(30, process_buffered_messages, args=(buffer_key,))
                        TIMER_BUFFER[buffer_key].start()
                    
                    return
                
                elif intent == 'follow_up':
                    # Customer following up on EXISTING request
                    logger.info(f"[SESSION] Customer following up on existing request")
                    
                    if ENABLE_FOLLOW_UP_RESPONSES:
                        # Send acknowledgment if enabled
                        acknowledgment = (
                            "Thank you for following up. "
                            "Our team has been notified and will contact you soon."
                        )
                        send_wassenger_reply(user_phone, acknowledgment, device_id, delay_seconds=1)
                        
                        # Notify sales team
                        notify_msg = (
                            f"📌 Follow-up from {user_phone}\n"
                            f"Message: {msg_text[:100]}...\n"
                            f"Previous service: {session.context.get('desired_service', 'unknown')}\n"
                            f"Closed {days_since_closed} days ago"
                        )
                        notify_sales_group(bot, notify_msg)
                    else:
                        # Silent tracking only
                        if not session.context:
                            session.context = {}
                        
                        follow_ups = session.context.get('silent_follow_ups', [])
                        follow_ups.append({
                            'timestamp': get_current_datetime_utc8().isoformat(),
                            'message': msg_text[:100]
                        })
                        session.context['silent_follow_ups'] = follow_ups[-5:]  # Keep last 5
                        db.session.commit()
                    
                    return
                
                else:  # intent == 'unclear'
                    # Can't determine intent - use safe approach
                    logger.info(f"[SESSION] Unclear intent - notifying sales team")
                    
                    # Track it
                    if not session.context:
                        session.context = {}
                    session.context['last_unclear_attempt'] = {
                        'timestamp': get_current_datetime_utc8().isoformat(),
                        'message': msg_text[:100]
                    }
                    db.session.commit()
                    
                    # Notify sales ONCE per day
                    notification_key = f"unclear:{user_phone}:{current_time_dt.date().isoformat()}"
                    notification_hash = hashlib.md5(notification_key.encode()).hexdigest()
                    
                    with BUFFER_LOCK:
                        if f"notif_{notification_hash}" not in MESSAGE_HASH_CACHE:
                            MESSAGE_HASH_CACHE[f"notif_{notification_hash}"] = time.time()
                            # Notify sales about unclear intent
                            notify_msg = (
                                f"⚠️ Customer {user_phone} sent message to closed session.\n"
                                f"Intent unclear - please review:\n"
                                f"Message: {msg_text[:100]}...\n"
                                f"Closed {days_since_closed} days ago"
                            )
                            notify_sales_group(bot, notify_msg)
                    
                    if ENABLE_FOLLOW_UP_RESPONSES:
                        # Send a generic response
                        response = (
                            "Thank you for your message. "
                            "Our team will review and contact you if needed."
                        )
                        send_wassenger_reply(user_phone, response, device_id, delay_seconds=1)
                    
                    return
            
            # === Normal flow for open sessions ===
            session_id = str(session.id)
            
            # Save incoming message
            save_message_safe(bot.id, user_phone, session_id, "in", msg_text, raw_media_url)
            
            # Add to buffer with limits
            buffer_key = (bot.id, user_phone, session_id)
            current_timestamp = time.time()
            
            with BUFFER_LOCK:
                if buffer_key not in BUFFER_START_TIME:
                    BUFFER_START_TIME[buffer_key] = current_timestamp
                
                MESSAGE_BUFFER[buffer_key].append({
                    "msg_text": msg_text,
                    "raw_media_url": raw_media_url,
                    "created_at": get_current_datetime_utc8().isoformat(),
                    "device_id": device_id,
                })
                
                buffer_age = current_timestamp - BUFFER_START_TIME[buffer_key]
                buffer_size = len(MESSAGE_BUFFER[buffer_key])
                
                if buffer_age > 60 or buffer_size >= 5:
                    if buffer_key in TIMER_BUFFER and TIMER_BUFFER[buffer_key]:
                        TIMER_BUFFER[buffer_key].cancel()
                    process_buffered_messages(buffer_key)
                else:
                    if buffer_key in TIMER_BUFFER and TIMER_BUFFER[buffer_key]:
                        TIMER_BUFFER[buffer_key].cancel()
                    TIMER_BUFFER[buffer_key] = Timer(30, process_buffered_messages, args=(buffer_key,))
                    TIMER_BUFFER[buffer_key].start()
                    
        except Exception as e:
            logger.error(f"[WEBHOOK ASYNC ERROR] {e}", exc_info=True)

# === MAIN WEBHOOK ENDPOINT ===
@app.route('/webhook', methods=['POST'])
def webhook():
    """Quick webhook response with async processing"""
    logger.info("[WEBHOOK] Received POST /webhook")
    data = request.json
    logger.info(f"[WEBHOOK] Incoming: {json.dumps(data)}")
    
    # Quick validation
    msg = data.get("data", {})
    if msg.get("flow") == "outbound":
        return jsonify({"status": "ignored"}), 200
    
    # Process in background
    thread = Thread(target=process_webhook_async, args=(data,))
    thread.daemon = True
    thread.start()
    
    # Return immediately
    return jsonify({"status": "queued"}), 200

# === HEALTH CHECK ENDPOINT ===
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test DB connection
        db.session.execute('SELECT 1')
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return jsonify({
        "status": "healthy",
        "buffered_messages": sum(len(msgs) for msgs in MESSAGE_BUFFER.values()),
        "active_timers": len(TIMER_BUFFER),
        "processing_flags": sum(1 for v in PROCESSING_FLAGS.values() if v),
        "db_status": db_status,
        "bypass_session_checks": BYPASS_SESSION_CHECKS,
        "enable_follow_up_responses": ENABLE_FOLLOW_UP_RESPONSES,
        "timestamp": get_current_datetime_utc8().isoformat()
    })

# === NEW: ADMIN ENDPOINT TO MANAGE SESSIONS ===
@app.route('/admin/session/<action>', methods=['POST'])
def admin_session(action):
    """Admin endpoint to manage sessions"""
    data = request.json
    user_phone = data.get("phone")
    bot_phone = data.get("bot_phone")
    
    if not user_phone or not bot_phone:
        return jsonify({"error": "Missing phone or bot_phone"}), 400
    
    bot = get_bot_by_phone(bot_phone)
    if not bot:
        return jsonify({"error": "Bot not found"}), 404
    
    customer = Customer.query.filter_by(phone_number=user_phone).first()
    if not customer:
        return jsonify({"error": "Customer not found"}), 404
    
    if action == "open":
        # Close existing and open new
        existing = Session.query.filter_by(
            customer_id=customer.id,
            bot_id=bot.id,
            status='open'
        ).first()
        if existing:
            existing.status = "closed"
            existing.ended_at = get_current_datetime_utc8()
        
        new_session = Session(
            customer_id=customer.id,
            bot_id=bot.id,
            started_at=get_current_datetime_utc8(),
            status='open',
            context={'admin_opened': True},
        )
        db.session.add(new_session)
        db.session.commit()
        
        return jsonify({"status": "opened", "session_id": new_session.id})
    
    elif action == "close":
        session = Session.query.filter_by(
            customer_id=customer.id,
            bot_id=bot.id,
            status='open'
        ).first()
        if session:
            session.status = "closed"
            session.ended_at = get_current_datetime_utc8()
            session.context['close_reason'] = "admin_closed"
            db.session.commit()
            return jsonify({"status": "closed", "session_id": session.id})
        else:
            return jsonify({"error": "No open session found"}), 404
    
    elif action == "status":
        session = Session.query.filter_by(
            customer_id=customer.id,
            bot_id=bot.id
        ).order_by(Session.started_at.desc()).first()
        
        if session:
            return jsonify({
                "session_id": session.id,
                "status": session.status,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "context": session.context
            })
        else:
            return jsonify({"error": "No session found"}), 404
    
    else:
        return jsonify({"error": "Invalid action"}), 400

# === CLEANUP FUNCTIONS ===
def cleanup_caches():
    """Clean up old cache entries to prevent memory leaks"""
    with BUFFER_LOCK:
        current_time = time.time()
        
        # Clean MESSAGE_HASH_CACHE
        keys_to_remove = []
        for key, timestamp in MESSAGE_HASH_CACHE.items():
            if current_time - timestamp > 3600:  # Remove entries older than 1 hour
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            MESSAGE_HASH_CACHE.pop(key, None)
        
        logger.info(f"[CLEANUP] Removed {len(keys_to_remove)} old cache entries")
        
        # Clean stuck buffers
        stuck_buffers = []
        for buffer_key, start_time in BUFFER_START_TIME.items():
            if current_time - start_time > 300:  # 5 minutes
                stuck_buffers.append(buffer_key)
        
        for buffer_key in stuck_buffers:
            logger.warning(f"[CLEANUP] Removing stuck buffer: {buffer_key}")
            MESSAGE_BUFFER.pop(buffer_key, None)
            BUFFER_START_TIME.pop(buffer_key, None)
            PROCESSING_FLAGS.pop(buffer_key, None)
            if buffer_key in TIMER_BUFFER:
                timer = TIMER_BUFFER.pop(buffer_key)
                if timer:
                    timer.cancel()
        
        logger.info(f"[CLEANUP] Removed {len(stuck_buffers)} stuck buffers")

def schedule_cleanup():
    """Schedule periodic cleanup"""
    cleanup_caches()
    # Schedule next cleanup in 1 hour
    cleanup_timer = Timer(3600, schedule_cleanup)
    cleanup_timer.daemon = True
    cleanup_timer.start()

# === MAIN RUN BLOCK ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure all tables exist
    
    # Start periodic cleanup
    schedule_cleanup()
    
    # Log configuration status
    logger.info(f"[CONFIG] BYPASS_SESSION_CHECKS: {BYPASS_SESSION_CHECKS}")
    logger.info(f"[CONFIG] ENABLE_FOLLOW_UP_RESPONSES: {ENABLE_FOLLOW_UP_RESPONSES}")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
