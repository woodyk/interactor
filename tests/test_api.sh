#!/usr/bin/env bash
#
# File: test_api.sh
# Author: Wadih Khairallah
# Description: 
# Created: 2025-05-18 21:56:25
# Modified: 2025-05-19 15:29:08

# Set your API key (for development mode, this key should work)
API_KEY="sk-interactor-api-dev"

curl -X GET "http://localhost:8000/v1/models" \
  -H "Authorization: Bearer $API_KEY"

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Hello, what can you do?"}
    ]
  }'

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Count from 1 to 10 slowly"}
    ],
    "stream": true
  }'

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "openai:gpt-4o",
    "messages": [
      {"role": "user", "content": "What is the weather in New York?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather in a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. New York, NY"
              }
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'

curl -X POST "http://localhost:8000/v1/sessions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "openai:gpt-4o-mini"
  }'

curl -X GET "http://localhost:8000/v1/sessions" \
  -H "Authorization: Bearer $API_KEY"

# Replace SESSION_ID with the ID from the create session response
SESSION_ID="session-12345"

curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"model\": \"openai:gpt-4o-mini\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Remember that my name is John\"}
    ],
    \"session_id\": \"$SESSION_ID\"
  }"

# Continue the conversation in the same session
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"model\": \"openai:gpt-4o-mini\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is my name?\"}
    ],
    \"session_id\": \"$SESSION_ID\"
  }"

curl -X DELETE "http://localhost:8000/v1/sessions/$SESSION_ID" \
  -H "Authorization: Bearer $API_KEY"

curl -X POST "http://localhost:8000/v1/system" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "prompt": "You are a helpful assistant that speaks like a pirate.",
    "session_id": "'"$SESSION_ID"'"
  }'

curl -X POST "http://localhost:8000/v1/functions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "name": "search_database",
    "description": "Search for information in a database",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query"
        },
        "database": {
          "type": "string",
          "enum": ["users", "products", "orders"],
          "description": "Database to search"
        }
      },
      "required": ["query"]
    }
  }'

curl -X GET "http://localhost:8000/v1/functions" \
  -H "Authorization: Bearer $API_KEY"

curl -X PUT "http://localhost:8000/v1/functions/search_database/disable" \
  -H "Authorization: Bearer $API_KEY"

curl -X PUT "http://localhost:8000/v1/functions/search_database/enable" \
  -H "Authorization: Bearer $API_KEY"

curl -X DELETE "http://localhost:8000/v1/functions/search_database" \
  -H "Authorization: Bearer $API_KEY"

curl -X GET "http://localhost:8000/v1/usage" \
  -H "Authorization: Bearer $API_KEY"

# Create a new session
SESSION_ID=$(curl -s -X POST "http://localhost:8000/v1/sessions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{}' | jq -r '.session_id')

echo "Created session: $SESSION_ID"

# Set a custom system prompt
curl -X POST "http://localhost:8000/v1/system" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"prompt\": \"You are an expert assistant with access to various tools. Be concise and helpful.\",
    \"session_id\": \"$SESSION_ID\"
  }"

# Register a function
curl -X POST "http://localhost:8000/v1/functions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "name": "calculate",
    "description": "Perform a calculation",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "The mathematical expression to evaluate"
        }
      },
      "required": ["expression"]
    }
  }'

# Start a conversation with a question that might trigger tool use
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"model\": \"openai:gpt-4o\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is the square root of 144 plus 50?\"}
    ],
    \"session_id\": \"$SESSION_ID\"
  }"

# Continue the conversation
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"model\": \"openai:gpt-4o\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What if we multiply that by 5?\"}
    ],
    \"session_id\": \"$SESSION_ID\"
  }"

# Check token usage
curl -X GET "http://localhost:8000/v1/usage" \
  -H "Authorization: Bearer $API_KEY"


