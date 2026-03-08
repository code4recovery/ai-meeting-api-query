#!/bin/bash

# --- CONFIGURATION ---
API_KEY=""
INPUT_DATA="meetings.json"
OUTPUT_FILE="final_meetings.json"
MODEL="gemini-2.5-flash"

echo "Checking for Sunday (day: 6) meetings..."

# --- STEP 1: JQ FILTER (The Truth) ---
# Extract all Sunday meetings into a temporary array
RAW_MATCHES=$(jq -c '[.[] | select(.day == 6 or .day == "6")]' "$INPUT_DATA")
MATCH_COUNT=$(echo "$RAW_MATCHES" | jq '. | length')

if [ "$MATCH_COUNT" -eq 0 ]; then
    echo "No Sunday meetings found."
    exit 1
fi

# Save the raw data for your records
echo "$RAW_MATCHES" | jq '.' > "$OUTPUT_FILE"
echo "Found $MATCH_COUNT meetings. Generating individual summaries..."
echo "----------------------------------------------"

# --- STEP 2: LOOP AND SUMMARIZE ---
for (( i=0; i<$MATCH_COUNT; i++ )); do
    # Extract specific fields for this specific meeting to keep the prompt "safe"
    NAME=$(echo "$RAW_MATCHES" | jq -r ".[$i].name")
    TIME=$(echo "$RAW_MATCHES" | jq -r ".[$i].time_formatted")
    LOC=$(echo "$RAW_MATCHES" | jq -r ".[$i].location")
    
    # Create the payload for this specific meeting
    cat <<EOF > payload.json
{
  "contents": [{
    "parts": [{
      "text": "Task: Create a short calendar summary. Name: $NAME. Time: $TIME. Location: $LOC. Format: [Name] at [Location] ([Time])."
    }]
  }],
  "safetySettings": [
    { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
    { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
    { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
    { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" }
  ]
}
EOF

    # Call Gemini
    RAW_RESPONSE=$(curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent?key=$API_KEY" \
      -H 'Content-Type: application/json' \
      -d @payload.json)

    # Extract the summary or fallback to a manual string if blocked
    SUMMARY=$(echo "$RAW_RESPONSE" | jq -r '.candidates[0].content.parts[0].text // empty')
    
    if [[ -z "$SUMMARY" || "$SUMMARY" == "null" ]]; then
        # Fallback: Manual string if the AI is still being picky
        SUMMARY="$NAME at $LOC ($TIME)"
    fi

    echo "Meeting $((i+1)): $SUMMARY"
done

echo "----------------------------------------------"
echo "Done! All data saved to $OUTPUT_FILE"

# Cleanup
rm -f payload.json#!/bin/bash

# --- CONFIGURATION ---
API_KEY=""
INPUT_DATA="meetings.json"
OUTPUT_FILE="final_meetings.json"
MODEL="gemini-2.5-flash"

echo "Checking for Sunday (day: 6) meetings..."

# --- STEP 1: JQ FILTER (The Truth) ---
# Extract all Sunday meetings into a temporary array
RAW_MATCHES=$(jq -c '[.[] | select(.day == 6 or .day == "6")]' "$INPUT_DATA")
MATCH_COUNT=$(echo "$RAW_MATCHES" | jq '. | length')

if [ "$MATCH_COUNT" -eq 0 ]; then
    echo "No Sunday meetings found."
    exit 1
fi

# Save the raw data for your records
echo "$RAW_MATCHES" | jq '.' > "$OUTPUT_FILE"
echo "Found $MATCH_COUNT meetings. Generating individual summaries..."
echo "----------------------------------------------"

# --- STEP 2: LOOP AND SUMMARIZE ---
for (( i=0; i<$MATCH_COUNT; i++ )); do
    # Extract specific fields for this specific meeting to keep the prompt "safe"
    NAME=$(echo "$RAW_MATCHES" | jq -r ".[$i].name")
    TIME=$(echo "$RAW_MATCHES" | jq -r ".[$i].time_formatted")
    LOC=$(echo "$RAW_MATCHES" | jq -r ".[$i].location")
    
    # Create the payload for this specific meeting
    cat <<EOF > payload.json
{
  "contents": [{
    "parts": [{
      "text": "Task: Create a short calendar summary. Name: $NAME. Time: $TIME. Location: $LOC. Format: [Name] at [Location] ([Time])."
    }]
  }],
  "safetySettings": [
    { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
    { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
    { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
    { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" }
  ]
}
EOF

    # Call Gemini
    RAW_RESPONSE=$(curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent?key=$API_KEY" \
      -H 'Content-Type: application/json' \
      -d @payload.json)

    # Extract the summary or fallback to a manual string if blocked
    SUMMARY=$(echo "$RAW_RESPONSE" | jq -r '.candidates[0].content.parts[0].text // empty')
    
    if [[ -z "$SUMMARY" || "$SUMMARY" == "null" ]]; then
        # Fallback: Manual string if the AI is still being picky
        SUMMARY="$NAME at $LOC ($TIME)"
    fi

    echo "Meeting $((i+1)): $SUMMARY"
done

echo "----------------------------------------------"
echo "Done! All data saved to $OUTPUT_FILE"

# Cleanup
rm -f payload.json
