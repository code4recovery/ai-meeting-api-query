# ai-meeting-api-query
Python3 app able to utilize AI for Generrative 12 Step Conversational Summaries, and API queries to produce json output & meeting query results to get query requests of keyword, lat, lon, and radius.

## Requires:
* linux webserver
* python3
* gemini API Key
* Google Maps API Key
* meetings.json or national-meetings.json
* python dependencies: numpy as np flask  Flask, request, jsonify, send_from_directory flask_cors CORS json re os math time traceback google genai

## Features
* from json meeting input it can respond to get query parameters, and produce TSML json output limited to the scope of the geolocation & keyword & radius query.

## QuickStart: 
1. replace DOMAIN with live domain in ensure_running.sh 
2. replace KEY with Google Maps API Key in frontend/index.html
3. create .env file with Gemini API Key

## Background:
* https://www.longbeachaa.org/matthew-l-naatw-ai-in-aa-new-depth-to-meeting-list-a/

## Examples:
* https://ai.lovethecode.cloud:5007/?q=renegades&lat=33.7799&lon=-118.328&radius=12
* https://ai.lovethecode.cloud:5012/api/ask?q=meeting&lat=33.7799&lon=-118.328&radius=12

