import os
import json
import psycopg2
import requests
from datetime import datetime, timedelta
import openai
import urllib.parse as up

# ✅ ENV CONFIG
DATABASE_URL = os.getenv("DATABASE_URL")
WASSENGER_API_KEY = os.getenv("WASSENGER_API_KEY")
DEVICE_ID = "6889d701d16d6f3e79e083c0"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MASTER_PHONE = "60127998080"

openai.api_key = OPENAI_API_KEY

# ✅ Parse DATABASE_URL for psycopg2
up.uses_netloc.append("postgres")
url = up.urlparse(DATABASE_URL)

DB_CONFIG = {
    "host": url.hostname,
    "port": url.port or 5432,
    "user": url.username,
    "password": url.password,
    "dbname": url.path[1:]
}

# ✅ WhatsApp send function
def send_whatsapp(phone, message):
    url = "https://api.wassenger.com/v1/messages"
    headers = {
        "Authorization": f"Bearer {WASSENGER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"phone": phone, "message": message, "device": DEVICE_ID}
    r = requests.post(url, headers=headers, json=payload)
    print(f"Sent to {phone}: {r.status_code}")
    return r.status_code == 200

# ✅ Fetch report data
def get_report(start_date, end_date, business_id=None):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    query = """
    SELECT 
        bus.id AS business_id,
        bus.name AS business_name,
        bt.id AS bot_id,
        bt.name AS bot_name,
        COUNT(*) AS total_sessions,
        COUNT(*) FILTER (WHERE s.context->>'close_reason' = 'won') AS total_won,
        COUNT(*) FILTER (WHERE s.context->>'close_reason' = 'lose') AS total_lose,
        ROUND(
            (COUNT(*) FILTER (WHERE s.context->>'close_reason' = 'won')::decimal 
             / NULLIF(COUNT(*), 0)) * 100, 2
        ) AS conversion_rate
    FROM session s
    JOIN bots bt ON s.bot_id = bt.id
    JOIN businesses bus ON bt.business_id = bus.id
    WHERE s.started_at BETWEEN %s AND %s
    """
    params = [start_date, end_date]

    if business_id:
        query += " AND bus.id = %s"
        params.append(business_id)

    query += " GROUP BY bus.id, bus.name, bt.id, bt.name ORDER BY bus.id, bt.id"
    cursor.execute(query, params)
    data = cursor.fetchall()

    cursor.execute("""
    SELECT s.context
    FROM session s
    JOIN bots bt ON s.bot_id = bt.id
    JOIN businesses bus ON bt.business_id = bus.id
    WHERE s.context->>'close_reason' = 'lose'
      AND s.started_at BETWEEN %s AND %s
    """ + (" AND bus.id = %s" if business_id else ""),
    params)
    lose_contexts = [json.dumps(row[0]) for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return data, lose_contexts

# ✅ GPT lose reason analysis
def analyze_lose_reasons(contexts):
    if not contexts:
        return "No lose sessions in this period."
    prompt = f"Analyze these lose reasons and summarize top causes:\n{contexts}"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a sales analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# ✅ Format report
def format_report(data, lose_analysis, start_date, end_date):
    report_msg = f"📊 Sales Report ({start_date.date()} → {end_date.date()})\n\n"
    current_business = None
    for business_id, business_name, bot_id, bot_name, total, won, lose, cr in data:
        if business_name != current_business:
            report_msg += f"🏢 {business_name}\n"
            current_business = business_name
        report_msg += f"  • {bot_name}: Total {total}, Won {won}, Lose {lose}, CR {cr or 0}%\n"
    report_msg += "\nTop Lose Reasons:\n" + lose_analysis
    return report_msg

# ✅ Weekly job
def send_weekly_reports():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, admin_phones FROM businesses;")
    businesses = cursor.fetchall()

    # ✅ Master report for 0127998080
    master_data, master_lose = get_report(start_date, end_date)
    master_analysis = analyze_lose_reasons(master_lose)
    master_message = format_report(master_data, master_analysis, start_date, end_date)
    send_whatsapp(MASTER_PHONE, master_message)

    # ✅ Per-business reports
    for business_id, admin_phones_json in businesses:
        # 🔧 Fix: handle both JSONB (list) and text
        if isinstance(admin_phones_json, list):
            admin_phones = admin_phones_json
        else:
            admin_phones = json.loads(admin_phones_json or '[]')

        if not admin_phones:
            continue

        data, lose_contexts = get_report(start_date, end_date, business_id)
        lose_analysis = analyze_lose_reasons(lose_contexts)
        message = format_report(data, lose_analysis, start_date, end_date)

        for phone in admin_phones:
            send_whatsapp(phone, message)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    send_weekly_reports()
