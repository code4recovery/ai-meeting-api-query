# ai-meeting-api-query
Python3 app able to utilize AI for Generrative 12 Step Conversational Summaries, and API queries to produce json output & meeting query results to get query requests of keyword, lat, lon, and radius.  

## Requires
* Linux webserver
* Python3
* Gemini API Key
* Google Maps API Key
* meetings.json or national-meetings.json
* python dependencies: numpy as np flask  Flask, request, jsonify, send_from_directory flask_cors CORS json re os math time traceback google genai

## Features
* from json meeting input it can respond to get query parameters, and produce TSML json output limited to the scope of the geolocation & keyword & radius query.

## QuickStart
1. replace KEY with Google Maps API Key in frontend/index.html
2. create .env file with Gemini API Key
3. run API Server, Configure aa-ai-meeting-finder-chat.html for Meeting Results

## Background:
* https://www.longbeachaa.org/matthew-l-naatw-ai-in-aa-new-depth-to-meeting-list-a/

## Examples:
* https://ragmg.matthews.help/ui
* https://ragmg.matthews.help/docs
* https://ragmg.matthews.help/health
* https://ai.lovethecode.cloud:5007/?q=renegades&lat=33.7799&lon=-118.328&radius=12
* https://ai.lovethecode.cloud:5012/api/ask?q=meeting&lat=33.7799&lon=-118.328&radius=12

## Other Prototypes to Demonstrate Features

### A. Python3 API Server/MongoDB TSML JSON/Gemini AI/Nginx Integrated Meeting Finder
* HTML5/CSS3/JavaScript client file (aa-ai-meeting-finder-chat.html) accesses API for Interactive Queries of Meetings.
* RAG from VectorStorage is utiized for Faster AI Query responses in updated Python3 API, To Be Released.

![Meeting Query AI Chat Summarizes](aameetingschat.png)
