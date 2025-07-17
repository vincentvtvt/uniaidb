import os
import logging
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

# ----- Flask & DB Setup -----
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/dbname")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ----- Model Definitions -----
class Agent(db.Model):
    __tablename__ = 'agents'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    agent_type = db.Column(db.String(30))
    config = db.Column(JSONB)
    active = db.Column(db.Boolean, default=True)

class BotTool(db.Model):
    __tablename__ = 'bot_tools'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    tool_id = db.Column(db.Integer)
    active = db.Column(db.Boolean, default=True)

class Bot(db.Model):
    __tablename__ = 'bots'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    phone_number = db.Column(db.String(30))
    config = db.Column(JSONB)
    created_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())
    system_prompt = db.Column(db.Text)
    manager_system_prompt = db.Column(db.Text)

class Customer(db.Model):
    __tablename__ = 'customers'
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(30))
    name = db.Column(db.String(50))
    language = db.Column(db.String(10))
    meta = db.Column(JSONB)
    created_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer)
    sender_type = db.Column(db.String(20))
    sender_id = db.Column(db.Integer)
    message = db.Column(db.Text)
    created_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())
    meta = db.Column(JSONB)
    payload = db.Column(JSONB)

class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer)
    bot_id = db.Column(db.Integer)
    started_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())
    ended_at = db.Column(db.TIMESTAMP(timezone=True))
    status = db.Column(db.String(20), default='open')
    goal = db.Column(db.String(50))
    context = db.Column(JSONB)

class Task(db.Model):
    __tablename__ = 'tasks'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer)
    customer_id = db.Column(db.Integer)
    agent_id = db.Column(db.Integer)
    type = db.Column(db.String(30))
    description = db.Column(db.Text)
    due_time = db.Column(db.TIMESTAMP(timezone=True))
    status = db.Column(db.String(20), default='pending')
    meta = db.Column(JSONB)
    created_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())

class Template(db.Model):
    __tablename__ = 'templates'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    template_id = db.Column(db.String(50))
    description = db.Column(db.String(255))
    content = db.Column(JSONB)
    language = db.Column(db.String(10))
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())
    updated_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now(), onupdate=func.now())

class Tool(db.Model):
    __tablename__ = 'tools'
    id = db.Column(db.Integer, primary_key=True)
    tool_id = db.Column(db.String(50))   # varchar(50), join with bot_tools.tool_id using CAST
    name = db.Column(db.String(100))
    description = db.Column(db.Text)
    prompt = db.Column(db.Text)
    action_type = db.Column(db.String(30))
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.TIMESTAMP(timezone=True), default=func.now())

class Workflow(db.Model):
    __tablename__ = 'workflows'
    id = db.Column(db.Integer, primary_key=True)
    bot_id = db.Column(db.Integer)
    agent_id = db.Column(db.Integer)
    name = db.Column(db.String(100))
    flow_config = db.Column(JSONB)
    active = db.Column(db.Boolean, default=True)

# ----- Example Utility: Get all active tools for a bot -----
def get_active_tools_for_bot(bot_id):
    """
    Returns a list of Tool objects that are active and assigned to the bot.
    Joins bot_tools (tool_id is int) and tools (tool_id is varchar) using CAST.
    """
    tools = (
        db.session.query(Tool)
        .join(BotTool, Tool.tool_id == db.cast(BotTool.tool_id, db.String))  # ensure both are string
        .filter(BotTool.bot_id == bot_id, BotTool.active == True, Tool.active == True)
        .all()
    )
    return tools

# ----- Example Endpoint: Webhook (expand as needed) -----
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    # Example: fetch bot_id from incoming data, get tools, etc.
    bot_id = data.get("bot_id")
    if not bot_id:
        return jsonify({"error": "No bot_id provided"}), 400

    tools = get_active_tools_for_bot(bot_id)
    tool_list = [
        {
            "id": t.id,
            "tool_id": t.tool_id,
            "name": t.name,
            "description": t.description,
            "prompt": t.prompt,
            "action_type": t.action_type,
        } for t in tools
    ]
    return jsonify({"tools": tool_list})

# ----- Main Entrypoint -----
if __name__ == "__main__":
    app.run(debug=True)
