# ai-meeting-api-query
Python3 app able to utilize AI for Generrative 12 Step Conversational Summaries, and API queries to produce json output & meeting query results to get query requests of keyword and conversational meeting queries.  

## Requires
* Linux webserver, Widows Server, also tested on Windows Desktop
* Python3
* Gemini API Key

## QuickStart
1. place in .env file your Gemini API Key
2. pip install -r requirements.txt
3. python service.py install #Installs as Windows Service for JSON responses from 127.0.0.1
4. python service.py start #Verify with curl -UseBasicParsing "http://127.0.0.1:8000/health"
5. query with $request = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/query" ` -ContentType "application/json" ` -Body '{"question": "What meetings are available in Seal Beach?"}'; $request.answer;

## Background:
* https://www.longbeachaa.org/matthew-l-naatw-ai-in-aa-new-depth-to-meeting-list-a/

## Examples:
* https://ragmg.matthews.help/ui
* https://ragmg.matthews.help/docs
* https://ragmg.matthews.help/health

![Meeting Query AI Chat Summarizes](aameetingschat.png)
