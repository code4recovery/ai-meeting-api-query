# ai-meeting-api-query
Python3 app able to utilize AI VectorStores to produce json output & meeting query results to get query requests of keyword, lat, lon, and radius.

Requires:
-linux webserver
-python3
-gemini API Key
-Google Maps API Key
-meetings.json or national-meetings.json

*from json meeting input it can respond to get query parameters, and produce TSML json output limited to the scope of the geolocation & keyword & radius query.
**python dependencies: numpy as np flask  Flask, request, jsonify, send_from_directory
flask_cors CORS json re os math time traceback google genai


ex: https://ai.lovethecode.cloud:5012/api/ask?q=meeting&lat=33.7799&lon=-118.328&radius=12
