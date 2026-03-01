import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import re
import os
import math
import time
import traceback
from google import genai

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

json_path = '/root/.openclaw/workspace/national-meetings.json'
meetings_data = []

def load_data():
    global meetings_data
    try:
        with open(json_path, 'r') as f:
            meetings_data = json.load(f)
        print(f"Successfully loaded {len(meetings_data)} meetings.")
    except Exception as e:
        print(f"Error loading meetings: {e}")

load_data()

client = None
if os.environ.get("GEMINI_API_KEY"):
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")

def haversine_distance(lat1, lon1, lat2, lon2):
    try:
        R = 3958.8
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    except:
        return 999999

def _generate_conversational_summary(query, results):
    if not results:
        return "No meetings found within 20 miles."
    if not client:
        return f"Found {len(results)} local meetings."
    try:
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        meetings_text = "\n".join([
            f"- {m.get('name')} ({m.get('time_formatted')}, {days[m.get('day', 0)] if isinstance(m.get('day'), int) else 'N/A'}) in {m.get('region')}."
            for m in results[:5]
        ])
        prompt = (
            f"User query: '{query}'\n"
            f"Results found: {len(results)}\n"
            f"Top matches:\n{meetings_text}\n\n"
            f"Write a 1-sentence friendly summary."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={'max_output_tokens': 100}
        )
        return response.text.strip()
    except Exception as e:
        print(f"AI Summary Error: {e}")
        return f"Found {len(results)} local meetings."

def get_answer(data):
    try:
        question = str(data.get('question', '')).lower()
        user_lat = data.get('latitude')
        user_lon = data.get('longitude')
        radius_miles = data.get('radius', 20) # Default to 20
        
        try:
            radius_miles = float(radius_miles)
        except:
            radius_miles = 20

        if user_lat is None or user_lon is None:
            return {"response_text": "Please provide location coordinates.", "match_count": 0, "matches": []}

        user_lat = float(user_lat)
        user_lon = float(user_lon)

        local_results = []
        # Calculate appropriate bounding box for dynamic radius
        lat_bound = radius_miles / 60.0
        lon_bound = radius_miles / 60.0

        for m in meetings_data:
            m_lat = m.get('latitude')
            m_lon = m.get('longitude')
            if m_lat is not None and m_lon is not None:
                try:
                    m_lat = float(m_lat)
                    m_lon = float(m_lon)
                    if abs(m_lat - user_lat) < lat_bound and abs(m_lon - user_lon) < lon_bound:
                        if haversine_distance(user_lat, user_lon, m_lat, m_lon) <= radius_miles:
                            local_results.append(m)
                except:
                    continue

        final_results = local_results
        if question.strip():
            query_words = question.split()
            final_results = []
            for m in local_results:
                search_blob = f"{m.get('name','') or ''} {m.get('location','') or ''} {m.get('region','') or ''} {m.get('notes','') or ''}".lower()
                if any(word in search_blob for word in query_words):
                    final_results.append(m)

        summary = _generate_conversational_summary(question, final_results)
        return {"response_text": summary, "match_count": len(final_results), "matches": final_results[:100]}
    except Exception as e:
        traceback.print_exc()
        return {"response_text": f"Internal Error: {str(e)}", "match_count": 0, "matches": []}

@app.route('/api/ask', methods=['POST', 'GET'])
def ask():
    if request.method == 'POST':
        data = request.json or {}
        # Identify if request is from the frontend (port 5007 proxy)
        # In this setup, we'll keep response_text for the frontend
        result = get_answer(data)
        return jsonify(result)
    else:
        # GET request: Assumed to be direct API call for port 5012
        data = {
            'question': request.args.get('q', ''),
            'latitude': request.args.get('lat'),
            'longitude': request.args.get('lon'),
            'radius': request.args.get('radius')
        }
        result = get_answer(data)
        return jsonify(result)

@app.route('/')
def serve_widget(): return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path): return send_from_directory('../frontend', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
