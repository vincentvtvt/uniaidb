import re
import logging
import time
import json
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
from threading import Timer
from collections import defaultdict
import anthropic


# Message buffer: {(bot_id, user_phone, session_id): [msg1, msg2, ...]}
MESSAGE_BUFFER = defaultdict(list)
TIMER_BUFFER = {}


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

class Lead(db.Model):
    __tablename__ = 'leads'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    business_id = db.Column(db.Integer)
    session_id = db.Column(db.String(50))
    name = db.Column(db.String(100), nullable=False)
    contact = db.Column(db.String(50), nullable=False)
    info = db.Column(db.JSON, default={})      # Flexible field for all extra info (dict)
    status = db.Column(db.String(20), default='open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

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

def download_to_bytes(url):
    """
    Download a file from a URL and return bytes.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content = resp.content
    logger.info(f"[DOWNLOAD] {len(content)} bytes downloaded from {url}. First 16 bytes: {content[:16]}")
    return content

def get_filename_from_url_or_path(input_value, default_ext=".pdf"):
    """
    Extracts filename from a URL or file path, or generates a random one.
    """
    if isinstance(input_value, str):
        base = os.path.basename(input_value.split("?")[0])
        if "." in base:
            return base
        else:
            return base + default_ext
    else:
        # If bytes or other, generate a random one
        return f"file-{uuid.uuid4().hex}{default_ext}"


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

def save_lead(
    name, contact, info_dict, bot_id=None, business_id=None, session_id=None, status="open"
):
    lead = Lead(
        name=name,
        contact=contact,
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
    # Guess file extension/type for filename and msg_type
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
        # Optionally guess filename if missing
        if isinstance(file_url_or_path, str) and not filename:
            filename = os.path.basename(file_url_or_path.split("?")[0])

    # Upload if not a file_id
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

def download_to_bytes(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

def upload_any_file_to_wassenger(file_path_or_bytes, filename=None, msg_type=None):
    """
    Uploads any file (PDF, image, etc.) to Wassenger via multipart/form-data.
    Accepts local file path, URL, or raw bytes. Returns file_id.
    Always sets a user-friendly or random filename.
    Logs and checks file signature before upload.
    """
    url = "https://api.wassenger.com/v1/files"
    headers = {"Token": WASSENGER_API_KEY}

    # Step 1: Get file content and filename
    if isinstance(file_path_or_bytes, str) and not file_path_or_bytes.startswith("http"):
        if not filename:
            filename = os.path.basename(file_path_or_bytes)
        with open(file_path_or_bytes, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_path_or_bytes, str) and file_path_or_bytes.startswith("http"):
        # Download file from URL
        file_bytes = download_to_bytes(file_path_or_bytes)
        if not filename:
            filename = get_filename_from_url_or_path(file_path_or_bytes, default_ext=".pdf" if msg_type == "media" else ".jpg")
    else:
        # Assume raw bytes (from memory), generate random filename if not provided
        file_bytes = file_path_or_bytes
        if not filename:
            ext = ".pdf" if msg_type == "media" else ".jpg"
            filename = f"file-{uuid.uuid4().hex}{ext}"

    # Debug: log file signature
    logger.debug(f"[UPLOAD DEBUG] filename: {filename}, size: {len(file_bytes)}, first 16 bytes: {file_bytes[:16]}")
    # Signature check (PDF, JPG, PNG)
    if filename.lower().endswith('.pdf') and not file_bytes.startswith(b'%PDF'):
        logger.error(f"[UPLOAD DEBUG] This is NOT a valid PDF file! (wrong header)")

    if filename.lower().endswith(('.jpg', '.jpeg')) and not file_bytes.startswith(b'\xff\xd8'):
        logger.warning(f"[UPLOAD DEBUG] This is NOT a valid JPG file! (wrong header)")

    if filename.lower().endswith('.png') and not file_bytes.startswith(b'\x89PNG'):
        logger.warning(f"[UPLOAD DEBUG] This is NOT a valid PNG file! (wrong header)")

    files = {"file": (filename, file_bytes)}

    # Step 2: Upload to Wassenger
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=30)
        if resp.status_code == 409:
            logger.warning(f"[MEDIA UPLOAD] Duplicate file detected (409 Conflict). {filename} already uploaded recently.")
            return None
        resp.raise_for_status()
        resp_json = resp.json()
        if isinstance(resp_json, list) and resp_json and resp_json[0].get('id'):
            file_id = resp_json[0]['id']
        elif isinstance(resp_json, dict) and resp_json.get('id'):
            file_id = resp_json['id']
        else:
            logger.error(f"[MEDIA UPLOAD FAIL] Wassenger /files bad response: {resp.text}")
            return None
        logger.info(f"[MEDIA UPLOAD SUCCESS] file_id: {file_id} for {filename}")
        return file_id
    except Exception as e:
        logger.error(f"[MEDIA UPLOAD FAIL] Wassenger /files error: {e}")
        return None


def send_wassenger_reply(phone, text, device_id, delay_seconds=0, msg_type="text", caption=None):
    """
    Always upload image/pdf/media to Wassenger unless text is already a file_id.
    - For "media" or "image": handles url, file path, or bytes (auto-upload)
    - For "text": sends as text
    """
    scheduled_time = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
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
        payload["deliverAt"] = scheduled_time

    elif msg_type in ("image", "media"):
        # Always upload unless text is already a file_id
        if isinstance(text, str) and len(text) == 24 and text.isalnum():
            # Wassenger file id, use directly
            payload["media"] = {"file": text}
        else:
            # Upload any URL, file path, or bytes and get file_id
            if isinstance(text, str) and text.startswith("http"):
                file_bytes = download_to_bytes(text)
                filename = "image.jpg" if msg_type == "image" else "document.pdf"
                file_id = upload_any_file_to_wassenger(file_bytes, filename=filename)
            else:
                file_id = upload_any_file_to_wassenger(text)
            if not file_id:
                logger.error(f"[SEND {msg_type.upper()}] Failed to upload to Wassenger")
                return None
            payload["media"] = {"file": file_id}
        if caption:
            payload["message"] = caption

    else:
        logger.error(f"Unsupported msg_type: {msg_type}")
        return None

    # Remove None values from payload
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

    # Build the tool menu and append to system prompt
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
    )
    
    logger.info(f"[AI DECISION] manager_system_prompt: {manager_prompt}")
    logger.info(f"[AI DECISION] history: {history_text}")
    
    try:
        # FIX 1: Use the manager_prompt as system message
        response = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.3,
            system=manager_prompt,  # Add this line!
            messages=[{"role": "user", "content": history_text}]
        )
        
        raw_response = response.content[0].text
        logger.info(f"[AI DECISION] Raw response: {raw_response}")
        
        # FIX 2: Extract reasoning first, then clean JSON
        reasoning_match = re.search(r'(.*?)(?=\{)', raw_response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            logger.info(f"[AI TOOL DECISION REASONING]: {reasoning}")
            print("\n[AI TOOL DECISION REASONING]:", reasoning)
        else:
            logger.info("[AI TOOL DECISION REASONING]: None found")

        # FIX 3: Clean and parse JSON more robustly
        json_block = strip_json_markdown_blocks(raw_response)
        
        # Try to extract JSON object from the response
        json_match = re.search(r'\{[^}]*"TOOLS"[^}]*\}', json_block)
        if json_match:
            json_str = json_match.group(0)
            tool_decision = json.loads(json_str)
            
            # Extract the TOOLS value
            tools_value = tool_decision.get("TOOLS")
            logger.info(f"[AI DECISION] Selected tool: {tools_value}")
            return tools_value
        else:
            logger.error(f"[AI DECISION] No valid JSON found in response: {json_block}")
            return "Default"
            
    except json.JSONDecodeError as e:
        logger.error(f"[AI DECISION] JSON decode error: {e}")
        logger.error(f"[AI DECISION] Raw response was: {raw_response}")
        return "Default"
    except Exception as e:
        logger.error(f"[AI DECISION] Unexpected error: {e}")
        return "Default"


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
    
    reply_prompt = build_json_prompt(prompt, example_json)
    logger.info(f"[AI REPLY] Prompt: {reply_prompt}")
    
    try:
        # FIX: Use system parameter correctly
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0.3,
            system=reply_prompt,  # Add this line!
            messages=[{"role": "user", "content": context_input}]
        ) as stream:
            reply_accum = ""
            print("[STREAM] Streaming model reply:")
            for text in stream.text_stream:
                reply_accum += text
                print(text, end="", flush=True)
            
            logger.info(f"\n[AI REPLY STREAMED]: {reply_accum}")
            
            # Clean and parse JSON response
            cleaned_response = strip_json_markdown_blocks(reply_accum)
            
            try:
                tool_decision = json.loads(cleaned_response)
                return tool_decision
            except json.JSONDecodeError as e:
                logger.error(f"[AI REPLY] JSON decode error: {e}")
                logger.error(f"[AI REPLY] Raw response was: {reply_accum}")
                # Return fallback response
                return {"message": ["I apologize, there was an error processing your request. Please try again."]}
                
    except Exception as e:
        logger.error(f"[AI REPLY] Unexpected error: {e}")
        return {"message": ["I apologize, there was an error processing your request. Please try again."]}

def process_ai_reply_and_send(customer_phone, ai_reply, device_id, bot_id=None, user=None, session_id=None):
    """
    Streams each message in ai_reply['message'] (if array) as separate WhatsApp messages,
    with short delays, and saves each outgoing message to DB.
    Handles special session-closing logic for notification/lead/closing.
    """

    def extract_field_from_notification(notification, field):
        if not notification:
            return None
        pattern = r'{field}[:：]\s*([^\n]+)'.format(field=re.escape(field))
        m = re.search(pattern, notification)
        return m.group(1).strip() if m else None

    try:
        parsed = ai_reply if isinstance(ai_reply, dict) else json.loads(ai_reply)
    except Exception as e:
        logger.error(f"[WEBHOOK] Could not parse AI reply as JSON: {ai_reply} ({e})")
        parsed = {}

    if not isinstance(parsed, dict):
        logger.error(f"[WEBHOOK] AI reply did not return a dict. Raw reply: {parsed}")
        parsed = {}

    # --- UNIVERSAL BACKEND CLOSING BLOCK ---
    if parsed.get("instruction") in ("close_session_and_notify_sales", "close_session_drop"):
        logger.info("[AI REPLY] Instruction: Session close detected. Executing closing/notification logic.")

        # 1. Save any extra fields to session context and close session
        info_to_save = {}
        for k, v in parsed.items():
            if k not in ("message", "notification", "instruction") and v is not None:
                info_to_save[k] = v

        close_reason = parsed.get("close_reason")
        if not close_reason:
            close_reason = "won" if parsed["instruction"] == "close_session_and_notify_sales" else "drop"

        # Try to capture specific drop/loss reason
        lose_reason = (
            parsed.get("lose_reason")
            or parsed.get("drop_reason")
            or info_to_save.get("lose_reason")
            or info_to_save.get("drop_reason")
        )
        if close_reason in ("drop", "lost", "lose") and lose_reason:
            close_reason = f"{close_reason}: {lose_reason}"

        # 2. Find and update the active session for this customer + bot
        bot = db.session.get(Bot, bot_id) if bot_id else None
        customer = Customer.query.filter_by(phone_number=customer_phone).first()

        if not customer:
            logger.error(f"[SESSION CLOSE] Customer not found for phone: {customer_phone}")
        if not bot:
            logger.error(f"[SESSION CLOSE] Bot not found for bot_id: {bot_id}")

        session_obj = None
        if bot and customer:
            session_obj = (
                db.session.query(Session)
                .filter_by(bot_id=bot.id, customer_id=customer.id, status="open")
                .order_by(Session.started_at.desc())
                .first()
            )
            if not session_obj:
                logger.error(f"[SESSION CLOSE] No open session found for bot_id={bot.id}, customer_id={customer.id}")
        else:
            session_obj = None
            logger.warning("[SESSION CLOSE] Could not look up session (bot or customer missing)")

        # --- Gather critical fields with fallback/defensive extraction ---
        critical_keys = ["name", "contact"]
        notification = parsed.get("notification") or info_to_save.get("notification") or ""

        name = parsed.get("name") or info_to_save.get("name") or extract_field_from_notification(notification, "Name")
        contact = (
            parsed.get("contact")
            or info_to_save.get("contact")
            or extract_field_from_notification(notification, "Contact")
            or extract_field_from_notification(notification, "Phone")
        )
        if not contact or str(contact).strip().lower() in [
            "whatsapp number", "[whatsapp number]", "same", "this", "use this", "ok", "yes"
        ]:
            contact = customer_phone

        info_fields = {}
        for k, v in {**parsed, **info_to_save}.items():
            if k not in critical_keys and v is not None:
                info_fields[k] = v

        if (
            parsed.get("instruction") == "close_session_and_notify_sales"
            and close_reason.startswith("won")
            and name and contact
        ):
            lead = Lead(
                name=name,
                contact=contact,
                info=info_fields,
                bot_id=bot_id,
                business_id=getattr(bot, 'business_id', None),
                session_id=session_obj.id if session_obj else None,
                status="open"
            )
            db.session.add(lead)
            db.session.commit()
            logger.info(f"[LEAD] Lead saved: {lead.id}, {lead.name}, {lead.contact}, {lead.info}")

            # Notify sales group if present
            if parsed.get("notification"):
                logger.info(f"[NOTIFY SALES GROUP]: {parsed['notification']}")
                notify_sales_group(bot, parsed["notification"])
        else:
            logger.warning(
                f"[LEAD] Not saving lead: missing required field(s) or not a win. "
                f"name={name}, contact={contact}, instruction={parsed.get('instruction')}, close_reason={close_reason}"
            )

        # --- Always send all customer-facing messages (up to 4) ---
        if "message" in parsed:
            msg_lines = parsed["message"]
            if isinstance(msg_lines, str):
                msg_lines = [msg_lines]
            for idx, part in enumerate(msg_lines[:4]):  # up to 4 messages
                if part:
                    delay = max(0, idx * 20)
                    send_wassenger_reply(customer_phone, part, device_id, delay_seconds=delay)
                    if bot_id and user and session_id:
                        save_message(bot_id, customer_phone, session_id, "out", part)

        return  # All done, prevents rest of function from running

    # --- Stream/send each message line-by-line (normal flow) ---
    if "message" in parsed and isinstance(parsed["message"], list):
        for idx, line in enumerate(parsed["message"]):
            delay = max(0, idx * 20)
            send_wassenger_reply(
                customer_phone,
                line,
                device_id,
                delay_seconds=delay,
                msg_type="text"
            )
            if bot_id and user and session_id:
                save_message(bot_id, customer_phone, session_id, "out", line)
    elif "message" in parsed:
        # fallback: send single message
        send_wassenger_reply(
            customer_phone,
            str(ai_reply),
            device_id,
            delay_seconds=5,
            msg_type="text"
        )
        if bot_id and user and session_id:
            save_message(bot_id, customer_phone, session_id, "out", str(ai_reply))

    # --- TEMPLATE PROCESSING ---
    if "template" in parsed:
        template_id = parsed["template"]
        template_content = get_template_content(template_id)
        doc_counter = 1
        img_counter = 1
        for idx, part in enumerate(template_content):
            content_type = part.get("type")
            content_value = part.get("content")
            caption = part.get("caption") or None
            delay = max(0, idx * 20)
            if content_type == "text":
                send_wassenger_reply(customer_phone, content_value, device_id, delay_seconds=delay)
                if bot_id and user and session_id:
                    save_message(bot_id, user, session_id, "out", content_value)
            elif content_type == "image":
                filename = f"image{img_counter}.jpg"
                img_counter += 1
                file_id = upload_media_file(content_value, db.session, filename=filename)
                if file_id:
                    send_wassenger_reply(
                        customer_phone,
                        file_id,
                        device_id,
                        msg_type="image",
                        caption=caption,
                        delay_seconds=delay
                    )
                    if bot_id and user and session_id:
                        save_message(bot_id, user, session_id, "out", "[Image sent]")
                else:
                    logger.warning(f"[MEDIA SEND] Failed to upload/send image: {filename}")
            elif content_type == "document":
                filename = f"document{doc_counter}.pdf"
                doc_counter += 1
                file_id = upload_media_file(content_value, db.session, filename=filename)
                if file_id:
                    send_wassenger_reply(
                        customer_phone,
                        file_id,
                        device_id,
                        msg_type="media",
                        caption=caption,
                        delay_seconds=delay
                    )
                    if bot_id and user and session_id:
                        save_message(bot_id, user, session_id, "out", "[PDF sent]")
                else:
                    logger.warning(f"[MEDIA SEND] Failed to upload/send document: {filename}")
        return  # All done, prevents rest of function from running

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

def process_buffered_messages(buffer_key):
    with app.app_context():
        bot_id, user_phone, session_id = buffer_key
        bot = db.session.get(Bot, bot_id)
        messages = MESSAGE_BUFFER.pop(buffer_key, [])
        if not messages:
            return
        device_id = messages[-1].get("device_id") 
        
        combined_text = "\n".join(m['msg_text'] for m in messages if m['msg_text'])
        history = get_latest_history(bot_id, user_phone, session_id)
        context_input = "\n".join([
            f"{'User' if m.direction == 'in' else 'Bot'}: {m.content}"
            for m in history
        ] + [f"User: {combined_text}"])
        tool_id = decide_tool_with_manager_prompt(bot, history)
        tool = None
        if tool_id and tool_id.lower() != "default":
            for t in get_active_tools_for_bot(bot.id):
                if t.tool_id == tool_id:
                    tool = t
                    break
        ai_reply = compose_reply(bot, tool, history, context_input)
        process_ai_reply_and_send(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)


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

                # ✅ ✅ ADD TRIGGER HANDLER HERE ✅ ✅
        if msg_text.strip() == "*.*":
            # Delete all previous conversation & restart session
            Message.query.filter_by(bot_id=bot.id, customer_phone=user_phone).delete()
            db.session.commit()
            # Close existing session & start a new one
            session.status = "closed"
            session.ended_at = datetime.now()
            db.session.commit()
            new_session = get_or_create_session(customer.id, bot.id)
            send_wassenger_reply(user_phone, "Convo Refreshed", device_id, delay_seconds=1)
            return jsonify({"status": "conversation_refreshed"}), 200

        if msg_text.strip().lower() == "*off*":
            # Force close session, do not reply to customer
            session.status = "closed"
            session.ended_at = datetime.now()
            session.context['close_reason'] = "force_closed"
            db.session.commit()
            logger.info(f"[SESSION] Force closed for {user_phone}")
            # Notify sales group (optional but recommended for audit trail)
            note = f"Session for {user_phone} was force closed by agent/trigger."
            notify_sales_group(bot, note)
            return jsonify({"status": "force_closed"}), 200
        # ✅ ✅ END OF TRIGGER HANDLER ✅ ✅

            # Add to message buffer
        # Add to message buffer and start/refresh the timer
        buffer_key = (bot.id, user_phone, session_id)
        MESSAGE_BUFFER[buffer_key].append({
            "msg_text": msg_text,
            "raw_media_url": raw_media_url,
            "created_at": datetime.now().isoformat(),
            "device_id": device_id,    # ADD THIS LINE
        })

        # --- Start or reset the 30s buffer timer for this key ---
        if buffer_key in TIMER_BUFFER and TIMER_BUFFER[buffer_key]:
            TIMER_BUFFER[buffer_key].cancel()
        TIMER_BUFFER[buffer_key] = Timer(30, process_buffered_messages, args=(buffer_key,))
        TIMER_BUFFER[buffer_key].start()

        return jsonify({"status": "buffered, will process in 30s"})

    

        
        # Buffer timer logic
        if buffer_key in TIMER_BUFFER and TIMER_BUFFER[buffer_key]:
            TIMER_BUFFER[buffer_key].cancel()  # Reset the timer
        TIMER_BUFFER[buffer_key] = Timer(30, process_buffered_messages, args=(buffer_key,))
        TIMER_BUFFER[buffer_key].start()

        return jsonify({"status": "buffered, will process in 30s"})

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


        # 8. Only send/process reply if session is NOT closed
        process_ai_reply_and_send(user_phone, ai_reply, device_id, bot_id=bot.id, user=user_phone, session_id=session_id)
        return jsonify({"status": "ok"})

    except Exception as e:
        logger.error(f"[WEBHOOK] Exception: {e}")
        return jsonify({"error": "Webhook processing error"}), 500

# 9. Main run block at file bottom only!
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
