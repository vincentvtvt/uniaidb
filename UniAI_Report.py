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

# Test customer IDs to exclude (configurable via env variable)
TEST_CUSTOMER_IDS = os.getenv("TEST_CUSTOMER_IDS", "7").split(",")
TEST_CUSTOMER_IDS = [int(id.strip()) for id in TEST_CUSTOMER_IDS if id.strip()]

# Define close reasons
WON_REASONS = ['won']
LOST_REASONS = ['drop']

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
    """Send WhatsApp message via Wassenger API"""
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
    """Generate report data for specified date range and optional business"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Build test customer exclusion clause
    exclude_test = ""
    exclude_params = []
    if TEST_CUSTOMER_IDS:
        placeholders = ','.join(['%s'] * len(TEST_CUSTOMER_IDS))
        exclude_test = f" AND s.customer_id NOT IN ({placeholders})"
        exclude_params = TEST_CUSTOMER_IDS
    
    query = f"""
    SELECT 
        bus.id AS business_id,
        bus.name AS business_name,
        bt.id AS bot_id,
        bt.name AS bot_name,
        COUNT(*) AS total_sessions,
        COUNT(*) FILTER (WHERE s.context->>'close_reason' = ANY(%s)) AS total_won,
        COUNT(*) FILTER (WHERE s.context->>'close_reason' = ANY(%s)) AS total_lost,
        COUNT(*) FILTER (WHERE s.context->>'close_reason' IS NULL) AS total_active,
        ROUND(
            (COUNT(*) FILTER (WHERE s.context->>'close_reason' = ANY(%s))::decimal 
             / NULLIF(COUNT(*) FILTER (WHERE s.context->>'close_reason' IS NOT NULL), 0)) * 100, 2
        ) AS conversion_rate
    FROM session s
    JOIN bots bt ON s.bot_id = bt.id
    JOIN businesses bus ON bt.business_id = bus.id
    WHERE s.started_at BETWEEN %s AND %s
    {exclude_test}
    """
    
    params = exclude_params + [WON_REASONS, LOST_REASONS, WON_REASONS, start_date, end_date]
    
    if business_id:
        query += " AND bus.id = %s"
        params.append(business_id)
    
    query += " GROUP BY bus.id, bus.name, bt.id, bt.name ORDER BY bus.id, bt.id"
    cursor.execute(query, params)
    data = cursor.fetchall()
    
    # Get lost session reasons with specific lose_reason field
    lose_query = f"""
    SELECT 
        s.context->>'close_reason' as reason_type,
        s.context->>'lose_reason' as specific_reason,
        bt.name as bot_name,
        bus.name as business_name
    FROM session s
    JOIN bots bt ON s.bot_id = bt.id
    JOIN businesses bus ON bt.business_id = bus.id
    WHERE s.context->>'close_reason' = ANY(%s)
      AND s.started_at BETWEEN %s AND %s
      AND s.context->>'lose_reason' IS NOT NULL
    {exclude_test}
    """
    
    lose_params = exclude_params + [LOST_REASONS, start_date, end_date]
    
    if business_id:
        lose_query += " AND bus.id = %s"
        lose_params.append(business_id)
    
    cursor.execute(lose_query, lose_params)
    lose_contexts = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return data, lose_contexts

# ✅ GPT lose reason analysis
def analyze_lose_reasons(contexts):
    """Analyze lose reasons using GPT-4o-mini"""
    if not contexts:
        return "✅ No lost sessions in this period - excellent performance!"
    
    # Group reasons by type
    reason_counts = {}
    specific_reasons = []
    bot_loses = {}
    
    for reason_type, specific_reason, bot_name, business_name in contexts:
        # Count by reason type
        reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        
        # Collect specific reasons
        if specific_reason:
            specific_reasons.append(specific_reason)
            
        # Track which bots are losing most
        bot_key = f"{business_name} - {bot_name}"
        bot_loses[bot_key] = bot_loses.get(bot_key, 0) + 1
    
    # Build analysis text
    analysis = f"📊 Total lost sessions: {len(contexts)}\n"
    
    # Top losing bots
    if bot_loses:
        top_bots = sorted(bot_loses.items(), key=lambda x: x[1], reverse=True)[:3]
        analysis += "\n🤖 Top bots with losses:\n"
        for bot, count in top_bots:
            analysis += f"  • {bot}: {count} losses\n"
    
    # Use GPT for deeper analysis if we have specific reasons
    if specific_reasons:
        # Take unique reasons to avoid repetition
        unique_reasons = list(set(specific_reasons))[:30]
        
        prompt = f"""Analyze these customer drop reasons and provide:
1. Top 3 patterns/themes
2. Brief actionable recommendation for each pattern

Drop reasons: {', '.join(unique_reasons)}

Be concise and practical."""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sales analyst specializing in chatbot optimization. Be concise and actionable."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            analysis += "\n💡 AI Analysis:\n" + response.choices[0].message["content"]
        except Exception as e:
            print(f"GPT analysis failed: {e}")
            # Fallback to simple frequency analysis
            from collections import Counter
            reason_freq = Counter(specific_reasons)
            top_3 = reason_freq.most_common(3)
            analysis += "\n📋 Top drop reasons:\n"
            for reason, count in top_3:
                analysis += f"  • {reason}: {count} times\n"
    
    return analysis

# ✅ Format report with better layout
def format_report(data, lose_analysis, start_date, end_date, is_master=False):
    """Format the report message for WhatsApp"""
    
    # Header
    report_type = "MASTER " if is_master else ""
    report_msg = f"📊 {report_type}Sales Report\n"
    report_msg += f"📅 {start_date.date()} → {end_date.date()}\n"
    report_msg += "=" * 30 + "\n\n"
    
    if not data:
        report_msg += "No session data for this period.\n"
        return report_msg
    
    # Calculate totals
    total_sessions = sum(row[4] for row in data)
    total_won = sum(row[5] for row in data)
    total_lost = sum(row[6] for row in data)
    total_active = sum(row[7] for row in data)
    overall_cr = round((total_won / total_sessions * 100) if total_sessions > 0 else 0, 2)
    
    # Overall summary
    report_msg += f"📈 OVERALL PERFORMANCE\n"
    report_msg += f"Total Sessions: {total_sessions}\n"
    report_msg += f"✅ Won: {total_won} | ❌ Lost: {total_lost} | ⏳ Active: {total_active}\n"
    report_msg += f"🎯 Overall CR: {overall_cr}%\n\n"
    
    # Per business/bot breakdown
    report_msg += "📊 BREAKDOWN BY BOT\n"
    report_msg += "-" * 25 + "\n"
    
    current_business = None
    for row in data:
        business_id, business_name, bot_id, bot_name, total, won, lost, active, cr = row
        
        if business_name != current_business:
            if current_business:  # Add spacing between businesses
                report_msg += "\n"
            report_msg += f"🏢 {business_name}\n"
            current_business = business_name
        
        # Calculate percentages
        won_pct = round((won/total * 100) if total > 0 else 0, 1)
        lost_pct = round((lost/total * 100) if total > 0 else 0, 1)
        
        report_msg += f"  📱 {bot_name}:\n"
        report_msg += f"     Sessions: {total}\n"
        report_msg += f"     ✅ Won: {won} ({won_pct}%)\n"
        report_msg += f"     ❌ Lost: {lost} ({lost_pct}%)\n"
        if active > 0:
            report_msg += f"     ⏳ Active: {active}\n"
        report_msg += f"     🎯 CR: {cr or 0}%\n"
    
    # Loss analysis section
    report_msg += "\n" + "=" * 30 + "\n"
    report_msg += "📉 LOSS ANALYSIS\n"
    report_msg += "-" * 25 + "\n"
    report_msg += lose_analysis
    
    # Footer with tips
    if total_lost > 0:
        report_msg += "\n\n💡 Quick Tips:\n"
        if overall_cr < 10:
            report_msg += "• Low CR detected - Review bot responses\n"
        if total_lost > total_won:
            report_msg += "• High drop rate - Check initial greeting\n"
        report_msg += "• Monitor peak drop times for patterns\n"
    
    # Test data notice if excluded
    if TEST_CUSTOMER_IDS:
        report_msg += f"\n📝 Note: Test data excluded (Customer IDs: {', '.join(map(str, TEST_CUSTOMER_IDS))})"
    
    return report_msg

# ✅ Weekly job
def send_weekly_reports():
    """Main function to send weekly reports"""
    print(f"Starting weekly report generation at {datetime.now()}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Get all businesses
    cursor.execute("SELECT id, name, admin_phones FROM businesses ORDER BY name;")
    businesses = cursor.fetchall()
    
    # ✅ Generate and send master report
    print("Generating master report...")
    master_data, master_lose = get_report(start_date, end_date)
    master_analysis = analyze_lose_reasons(master_lose)
    master_message = format_report(master_data, master_analysis, start_date, end_date, is_master=True)
    
    if send_whatsapp(MASTER_PHONE, master_message):
        print(f"✅ Master report sent to {MASTER_PHONE}")
    else:
        print(f"❌ Failed to send master report to {MASTER_PHONE}")
    
    # ✅ Per-business reports
    success_count = 0
    fail_count = 0
    
    for business_id, business_name, admin_phones_json in businesses:
        print(f"\nProcessing {business_name}...")
        
        # Handle both JSONB (list) and text formats
        if isinstance(admin_phones_json, list):
            admin_phones = admin_phones_json
        else:
            try:
                admin_phones = json.loads(admin_phones_json or '[]')
            except json.JSONDecodeError:
                print(f"  ⚠️ Invalid admin_phones format for {business_name}")
                continue
        
        if not admin_phones:
            print(f"  ⚠️ No admin phones configured for {business_name}")
            continue
        
        # Generate business-specific report
        data, lose_contexts = get_report(start_date, end_date, business_id)
        
        if not data:
            print(f"  ℹ️ No data for {business_name} this week")
            continue
        
        lose_analysis = analyze_lose_reasons(lose_contexts)
        message = format_report(data, lose_analysis, start_date, end_date, is_master=False)
        
        # Send to all admin phones
        for phone in admin_phones:
            if send_whatsapp(phone, message):
                print(f"  ✅ Report sent to {phone}")
                success_count += 1
            else:
                print(f"  ❌ Failed to send to {phone}")
                fail_count += 1
    
    cursor.close()
    conn.close()
    
    # Final summary
    print("\n" + "=" * 40)
    print(f"Report generation completed at {datetime.now()}")
    print(f"✅ Successful sends: {success_count + 1}")  # +1 for master
    print(f"❌ Failed sends: {fail_count}")
    print(f"📊 Total businesses processed: {len(businesses)}")
    
    return success_count, fail_count

# ✅ Main execution
if __name__ == "__main__":
    try:
        send_weekly_reports()
    except Exception as e:
        print(f"Critical error: {e}")
        # Send error notification to master phone
        error_msg = f"⚠️ Report Generation Failed\n\nError: {str(e)}\nTime: {datetime.now()}"
        send_whatsapp(MASTER_PHONE, error_msg)
