#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: api.py
# Description: REST API for Universal AI interaction toolkit
#              OpenAI-compatible with extended features
# Created: 2025-05-18
# Modified: 2025-05-19 21:21:56

import os
import json
import uuid
import time
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

# FastAPI imports for API framework
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator, create_model

# Import the Interactor class - handle both package and direct imports
import sys
import os

# Set up console output
from rich.console import Console
console = Console()
log = console.log# Add CORS middleware


# Add parent directory to path if running as a standalone script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from interactor import Interactor
except ImportError:
    # Try relative import if within package
    try:
        from ..interactor import Interactor
    except ImportError:
        # Last resort - try direct import assuming current directory structure
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from interactor import Interactor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai-api")

# ----------------------
# Server lifecycle context manager
# ----------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """Initialize server on startup and clean up on shutdown."""
    # Startup logic
    # Load API keys from environment
    api_keys_env = os.getenv("API_KEYS", "")
    if api_keys_env:
        keys = [key.strip() for key in api_keys_env.split(",")]
        for key in keys:
            if key:
                API_KEYS[key] = True
    
    # In development mode, add a default key if none configured
    if not API_KEYS and os.getenv("DEV_MODE", "false").lower() == "true":
        default_key = "sk-interactor-api-dev"
        API_KEYS[default_key] = True
        logger.warning(f"DEV MODE: Using default API key: {default_key}")
    
    logger.info(f"API initialized with {len(API_KEYS)} API keys")
    
    yield
    
    # Shutdown logic
    # Clean up any resources here if needed
    logger.info("API server shutting down")

# Recreate FastAPI app with lifespan context manager
app = FastAPI(
    title="Universal AI Interaction API",
    description="OpenAI-compatible REST API with extended features for AI interactions",
    version="1.0.0",
    lifespan=lifespan
)

# API key security
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Global settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai:gpt-4o-mini")
API_KEYS = {"sk-interactor-api-dev"}  # Store allowed API keys
INTERACTORS = {}  # Store interactor instances by user_id or api_key
SESSIONS = {}  # Store active sessions

# ----------------------
# Pydantic models for API
# ----------------------

# OpenAI-compatible models
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    session_id: Optional[str] = None
    base_url: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields for future compatibility

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# Extended API models
class SystemPromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class SystemPromptResponse(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    model: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    token_count: int
    model: str

class AvailableModelsResponse(BaseModel):
    models: List[str]
    default_model: str

class TokenUsageResponse(BaseModel):
    total: int
    limit: int
    percentage: float
    by_role: Dict[str, int]
    current_model: str
    current_provider: str

class FunctionRegistrationRequest(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    override: Optional[bool] = False
    disabled: Optional[bool] = False

class FunctionResponse(BaseModel):
    name: str
    status: str
    message: Optional[str] = None

class FunctionListResponse(BaseModel):
    functions: List[Dict[str, Any]]

# ----------------------
# Authentication and session handling
# ----------------------

async def get_api_key(request: Request, auth_header: Optional[str] = Depends(api_key_header)):
    """Extract and validate API key from Authorization header."""
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Handle 'Bearer' prefix if present
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    else:
        api_key = auth_header
        
    # For demo/development - if no keys are configured, allow any key
    if not API_KEYS and os.getenv("DEV_MODE", "false").lower() == "true":
        return api_key
    
    # For production - validate the key
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

def get_interactor(api_key: str = Depends(get_api_key)):
    """Get or create an Interactor instance for the API key."""
    if api_key not in INTERACTORS:
        # Create a new Interactor instance for this API key
        INTERACTORS[api_key] = Interactor(
            model=DEFAULT_MODEL,
            stream=True,  # Default to streaming support
            tools=True,   # Enable tools support by default
            session_enabled=False,  # Enable session support
        )
        logger.info(f"Created new Interactor for key ending with ...{api_key[-4:]}")
    
    return INTERACTORS[api_key]

# ----------------------
# Helper functions
# ----------------------

def create_chat_completion_response(interactor, response_content, request_data):
    """Create an OpenAI-compatible completion response."""
    # Generate a unique ID for the completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    # Get token usage information
    token_usage = interactor.track_token_usage()
    
    # Estimate completion tokens using the last content
    content_tokens = interactor._count_tokens([{"role": "assistant", "content": response_content}])
    
    # Calculate prompt tokens (current total - completion)
    prompt_tokens = token_usage["current"] - content_tokens
    
    # Create the response object
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_data.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": content_tokens,
            "total_tokens": prompt_tokens + content_tokens
        }
    }
    
    return response

def convert_openai_tools_to_interactor(tools_data):
    """Convert OpenAI-format tools to Interactor's format."""
    if not tools_data:
        return []
    
    converted_tools = []
    for tool in tools_data:
        try:
            # Check if it's already in the expected format
            if "type" in tool and tool["type"] == "function" and "function" in tool:
                converted_tools.append(tool)
            else:
                # Convert to expected format
                converted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                }
                converted_tools.append(converted_tool)
        except Exception as e:
            logger.warning(f"Error converting tool: {e}")
            continue
    
    return converted_tools

def convert_messages_to_interactor_format(messages):
    """Convert API message format to Interactor's internal format."""
    converted = []
    for msg in messages:
        try:
            if msg.role == "system":
                # System messages are handled separately in Interactor
                continue
                
            converted_msg = {
                "role": msg.role,
                "content": msg.content or ""
            }
            
            # Handle tool-related messages
            if msg.tool_calls:
                converted_msg["tool_calls"] = msg.tool_calls
                
            if msg.tool_call_id:
                converted_msg["tool_call_id"] = msg.tool_call_id
                
            converted.append(converted_msg)
        except Exception as e:
            logger.warning(f"Error converting message: {e}")
            continue
    
    return converted

# ----------------------
# API Endpoints - OpenAI Compatible
# ----------------------

@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    interactor: Interactor = Depends(get_interactor)
):
    """Create a chat completion - OpenAI compatible endpoint."""
    try:
        # Configure interactor based on request
        if request_data.model:
            # Only switch model if different from current
            if f"{interactor.provider}:{interactor.model}" != request_data.model:
                interactor._setup_client(request_data.model, base_url=request_data.base_url)
                interactor._setup_encoding()

        # Handle system message if present
        system_messages = [msg for msg in request_data.messages if msg.role == "system"]
        if system_messages:
            # Use the last system message
            interactor.messages_system(system_messages[-1].content)

        # Prepare session
        session_id = request_data.session_id

        # Add user messages (excluding system) to the history
        for msg in request_data.messages:
            if msg.role != "system":  # Skip system, already handled
                interactor.messages_add(role=msg.role, content=msg.content or "")

        # Set up tools if provided
        if request_data.tools:
            # Convert tools to interactor format and register them
            for tool in convert_openai_tools_to_interactor(request_data.tools):
                if "function" in tool:
                    func = tool["function"]
                    # Register as placeholder function that returns arguments
                    def placeholder_func(**kwargs):
                        return {"arguments": kwargs, "status": "called"}

                    # Add to interactor with override=True to avoid errors
                    interactor.add_function(
                        placeholder_func,
                        name=func.get("name"),
                        description=func.get("description", ""),
                        override=True
                    )

        # Handle streaming responses
        if request_data.stream:
            # Return a streaming response using our async generator
            return StreamingResponse(
                stream_chat_completion(interactor, request_data, session_id),
                media_type="text/event-stream"
            )
        else:
            # Get last user message for non-streaming completion
            user_msg = next((msg.content for msg in reversed(request_data.messages)
                            if msg.role == "user"), "")

            # If no user message found, use an empty string
            if not user_msg and request_data.messages:
                logger.warning("No user message found in the request, using empty string")
                user_msg = ""

            # Use _interact_async_core directly for async handling
            response_content = await interactor._interact_async_core(
                user_input=user_msg,
                quiet=True,
                tools=True if request_data.tools else False,
                stream=False,
                markdown=False
            )

            # Create OpenAI-compatible response
            response = create_chat_completion_response(interactor, response_content, request_data)
            return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat completion: {str(e)}"
        )


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    request_data: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    interactor: Interactor = Depends(get_interactor)
):
    """Create a chat completion - OpenAI compatible endpoint."""
    try:
        # Configure interactor based on request
        if request_data.model:
            # Only switch model if different from current
            if f"{interactor.provider}:{interactor.model}" != request_data.model:
                interactor._setup_client(request_data.model, base_url=request_data.base_url)
                interactor._setup_encoding()
        
        # Handle system message if present
        system_messages = [msg for msg in request_data.messages if msg.role == "system"]
        if system_messages:
            # Use the last system message
            interactor.messages_system(system_messages[-1].content)
            
        # Prepare session
        session_id = request_data.session_id
        
        # Add user messages (excluding system) to the history
        for msg in request_data.messages:
            if msg.role != "system":  # Skip system, already handled
                interactor.messages_add(role=msg.role, content=msg.content or "")
        
        # Set up tools if provided
        if request_data.tools:
            # Convert tools to interactor format and register them
            for tool in convert_openai_tools_to_interactor(request_data.tools):
                if "function" in tool:
                    func = tool["function"]
                    # Register as placeholder function that returns arguments
                    def placeholder_func(**kwargs):
                        return {"arguments": kwargs, "status": "called"}
                    
                    # Add to interactor with override=True to avoid errors
                    interactor.add_function(
                        placeholder_func,
                        name=func.get("name"),
                        description=func.get("description", ""),
                        override=True
                    )
        
        # Handle streaming responses
        if request_data.stream:
            # Return a streaming response
            return StreamingResponse(
                stream_chat_completion(interactor, request_data, session_id),
                media_type="text/event-stream"
            )
        else:
            # Get last user message for non-streaming completion
            user_msg = next((msg.content for msg in reversed(request_data.messages) 
                            if msg.role == "user"), "")
            
            # If no user message found, use an empty string
            if not user_msg and request_data.messages:
                logger.warning("No user message found in the request, using empty string")
                user_msg = ""
            
            # We must avoid using asyncio.run() inside an existing event loop
            # Modified version of interact that uses the current event loop
            loop = asyncio.get_event_loop()
            
            # Use _interact_async_core directly for async handling
            response_content = await interactor._interact_async_core(
                user_input=user_msg,
                quiet=True,
                tools=True if request_data.tools else False,
                stream=False,
                markdown=False
            )
            
            # Create OpenAI-compatible response
            response = create_chat_completion_response(interactor, response_content, request_data)
            return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat completion: {str(e)}"
        )

@app.get("/v1/models", response_model=AvailableModelsResponse)
async def list_models(interactor: Interactor = Depends(get_interactor)):
    """List available models - OpenAI compatible endpoint."""
    try:
        # Force update of available models
        models = interactor.list_models(update=True)
        
        return {
            "models": models,
            "default_model": DEFAULT_MODEL
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

# ----------------------
# API Endpoints - Extended Features
# ----------------------

@app.post("/v1/system", response_model=SystemPromptResponse)
async def set_system_prompt(
    request_data: SystemPromptRequest,
    interactor: Interactor = Depends(get_interactor)
):
    """Set the system prompt for future interactions."""
    try:
        # Set the system prompt
        updated_prompt = interactor.messages_system(request_data.prompt)
        
        # Handle session if provided
        if request_data.session_id:
            interactor.session_load(request_data.session_id)
        
        return {
            "prompt": updated_prompt,
            "session_id": request_data.session_id
        }
    except Exception as e:
        logger.error(f"Error setting system prompt: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error setting system prompt: {str(e)}"
        )

@app.post("/v1/sessions", response_model=SessionResponse)
async def create_or_load_session(
    request_data: SessionRequest,
    interactor: Interactor = Depends(get_interactor)
):
    """Create a new session or load an existing one."""
    try:
        # Generate a session ID if not provided
        session_id = request_data.session_id or f"session-{uuid.uuid4().hex}"
        
        # Switch model if requested
        if request_data.model:
            interactor._setup_client(request_data.model)
            interactor._setup_encoding()
        
        # Load or create the session
        interactor.session_load(session_id)
        
        # Get session info
        message_count = len(interactor.messages())
        token_count = interactor.track_token_usage()["current"]
        
        return {
            "session_id": session_id,
            "message_count": message_count,
            "token_count": token_count,
            "model": f"{interactor.provider}:{interactor.model}"
        }
    except Exception as e:
        logger.error(f"Error creating/loading session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating/loading session: {str(e)}"
        )

@app.delete("/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    interactor: Interactor = Depends(get_interactor)
):
    """Reset or delete a session."""
    try:
        # Check if this is the active session
        if interactor.session_id == session_id:
            # Reset the session
            interactor.session_reset()
        
        # If session storage is used, delete the session file
        if interactor.session:
            interactor.session.delete(session_id)
        
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )

@app.get("/v1/sessions", response_model=Dict[str, Any])
async def list_sessions(interactor: Interactor = Depends(get_interactor)):
    """List all available sessions."""
    try:
        sessions = []
        
        # Get sessions from the session manager
        if interactor.session:
            session_list = interactor.session.list()
            for session_id in session_list:
                # Load basic session info
                try:
                    session_data = interactor.session.load_meta(session_id)
                    sessions.append({
                        "session_id": session_id,
                        "created": session_data.get("created", "unknown"),
                        "updated": session_data.get("updated", "unknown"),
                        "message_count": len(session_data.get("messages", [])),
                    })
                except Exception as e:
                    # Skip sessions with errors
                    logger.warning(f"Error loading session {session_id}: {str(e)}")
        
        return {
            "sessions": sessions,
            "active_session": interactor.session_id,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sessions: {str(e)}"
        )

@app.get("/v1/usage", response_model=TokenUsageResponse)
async def get_token_usage(interactor: Interactor = Depends(get_interactor)):
    """Get token usage statistics for the current session."""
    try:
        # Get comprehensive token breakdown
        breakdown = interactor.get_message_token_breakdown()
        
        # Get overall usage stats
        usage = interactor.track_token_usage()
        
        return {
            "total": usage["current"],
            "limit": usage["limit"],
            "percentage": usage["percentage"],
            "by_role": breakdown["by_role"],
            "current_model": interactor.model,
            "current_provider": interactor.provider
        }
    except Exception as e:
        logger.error(f"Error getting token usage: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting token usage: {str(e)}"
        )

@app.post("/v1/functions", response_model=FunctionResponse)
async def register_function(
    request_data: FunctionRegistrationRequest,
    interactor: Interactor = Depends(get_interactor)
):
    """Register a new tool function."""
    try:
        # Check if function already exists
        existing_functions = interactor.list_functions()
        function_exists = any(f["function"]["name"] == request_data.name for f in existing_functions)
        
        # Automatically set override to True if function exists
        should_override = request_data.override or function_exists
        
        # Create a placeholder function that returns its arguments
        def placeholder_function(**kwargs):
            return {
                "name": request_data.name,
                "arguments": kwargs,
                "status": "called"
            }
        
        # Register the function with Interactor
        interactor.add_function(
            placeholder_function,
            name=request_data.name,
            description=request_data.description,
            override=should_override
        )
        
        return {
            "name": request_data.name,
            "status": "registered" if not function_exists else "updated",
            "message": "Function registered successfully" if not function_exists else "Function updated successfully"
        }
    except Exception as e:
        logger.error(f"Error registering function: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error registering function: {str(e)}"
        )

@app.get("/v1/functions", response_model=FunctionListResponse)
async def list_functions(interactor: Interactor = Depends(get_interactor)):
    """List all registered functions."""
    try:
        functions = interactor.list_functions()
        return {"functions": functions}
    except Exception as e:
        logger.error(f"Error listing functions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing functions: {str(e)}"
        )

@app.put("/v1/functions/{function_name}/enable")
async def enable_function(
    function_name: str,
    interactor: Interactor = Depends(get_interactor)
):
    """Enable a previously disabled function."""
    try:
        success = interactor.enable_function(function_name)
        if success:
            return {"status": "enabled", "name": function_name}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Function {function_name} not found"
            )
    except Exception as e:
        logger.error(f"Error enabling function: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error enabling function: {str(e)}"
        )

@app.put("/v1/functions/{function_name}/disable")
async def disable_function(
    function_name: str,
    interactor: Interactor = Depends(get_interactor)
):
    """Disable a function without removing it."""
    try:
        success = interactor.disable_function(function_name)
        if success:
            return {"status": "disabled", "name": function_name}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Function {function_name} not found"
            )
    except Exception as e:
        logger.error(f"Error disabling function: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error disabling function: {str(e)}"
        )

@app.delete("/v1/functions/{function_name}")
async def delete_function(
    function_name: str,
    interactor: Interactor = Depends(get_interactor)
):
    """Delete a registered function."""
    try:
        success = interactor.delete_function(function_name)
        if success:
            return {"status": "deleted", "name": function_name}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Function {function_name} not found"
            )
    except Exception as e:
        logger.error(f"Error deleting function: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting function: {str(e)}"
        )

async def stream_chat_completion(interactor, request_data, session_id=None):
    """Generate streaming completion chunks in OpenAI format.

    This function returns an async generator that yields SSE-formatted chunks
    for a streaming chat completion.

    Args:
        interactor: The Interactor instance
        request_data: The chat completion request data
        session_id: Optional session ID

    Returns:
        An async generator yielding SSE-formatted chunks
    """
    # Create a unique ID for this completion
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_timestamp = int(time.time())

    # Extract user message from the request
    user_msg = next((msg.content for msg in reversed(request_data.messages)
                     if msg.role == "user"), "")

    # If no user message found, use empty string
    if not user_msg and request_data.messages:
        logger.warning("No user message found in the request, using empty string")
        user_msg = ""

    # Send initial chunk with empty delta
    yield f"data: {json.dumps({
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': created_timestamp,
        'model': request_data.model,
        'choices': [
            {
                'index': 0,
                'delta': {},
                'finish_reason': None
            }
        ]
    })}\n\n"

    # Setup a queue for the streaming output
    chunk_queue = asyncio.Queue()
    finished = False

    # Create a non-async wrapper for the callback to handle sync/async compatibility
    def output_callback(text):
        """Wrapper for handling callback from both sync and async contexts.

        This is a synchronous function that schedules the async processing
        of a text chunk without awaiting it, avoiding the coroutine not awaited warning.
        """
        async def process_chunk_async(chunk_text):
            # Handle tool call status messages (JSON format)
            if isinstance(chunk_text, str) and chunk_text.startswith("{") and chunk_text.endswith("}"):
                try:
                    tool_data = json.loads(chunk_text)
                    if tool_data.get("type") == "tool_call":
                        # Send tool status info in a metadata chunk
                        tool_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_timestamp,
                            "model": request_data.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_data.get("tool_name"),
                                                    "arguments": "{}"
                                                }
                                            }
                                        ]
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        await chunk_queue.put(f"data: {json.dumps(tool_chunk)}\n\n")
                        return
                except (json.JSONDecodeError, TypeError):
                    pass  # Not JSON or wrong format, treat as normal content

            # Process normal content chunk
            if chunk_text:
                content_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request_data.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk_text
                            },
                            "finish_reason": None
                        }
                    ]
                }
                await chunk_queue.put(f"data: {json.dumps(content_chunk)}\n\n")

        # Schedule the async task without awaiting it
        asyncio.create_task(process_chunk_async(text))

    # Start the interaction in the background
    async def run_interaction():
        nonlocal finished
        try:
            # Use _interact_async_core directly to avoid asyncio.run() issues
            await interactor._interact_async_core(
                user_input=user_msg,
                quiet=True,
                tools=True if request_data.tools else False,
                stream=True,
                markdown=False,
                output_callback=output_callback
            )
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}", exc_info=True)
            # Add error information to the queue
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request_data.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error"
                    }
                ],
                "error": {
                    "message": str(e),
                    "type": "api_error"
                }
            }
            await chunk_queue.put(f"data: {json.dumps(error_chunk)}\n\n")
        finally:
            # Signal that we're done sending chunks
            finished = True
            # Wait a short time for any remaining chunks to be processed
            await asyncio.sleep(0.5)
            # Add final chunk and done marker
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request_data.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            await chunk_queue.put(f"data: {json.dumps(final_chunk)}\n\n")
            await chunk_queue.put("[DONE]")

    # Start the interaction task
    interaction_task = asyncio.create_task(run_interaction())

    # Yield chunks as they become available
    try:
        while True:
            # Wait for chunks with timeout
            try:
                chunk = await asyncio.wait_for(chunk_queue.get(), 30.0)

                # Check if we've reached the end
                if chunk == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                # Yield the chunk
                yield chunk

            except asyncio.TimeoutError:
                # If we timeout and the interaction is still running, continue waiting
                if not finished and not interaction_task.done():
                    logger.warning("Timeout waiting for chunk, but interaction still running")
                    continue

                # If interaction is done but we timed out, something went wrong
                if finished or interaction_task.done():
                    logger.error("Timeout and interaction complete - possible missed chunks")
                    # Send a final chunk and done marker
                    yield f"data: {json.dumps({
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created_timestamp,
                        'model': request_data.model,
                        'choices': [
                            {
                                'index': 0,
                                'delta': {},
                                'finish_reason': 'timeout'
                            }
                        ]
                    })}\n\n"
                    yield "data: [DONE]\n\n"
                    break
    except Exception as e:
        logger.error(f"Error in stream generator: {str(e)}", exc_info=True)
        # Send error information
        yield f"data: {json.dumps({
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': created_timestamp,
            'model': request_data.model,
            'choices': [
                {
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'error'
                }
            ],
            'error': {
                'message': str(e),
                'type': 'api_error'
            }
        })}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Ensure the interaction task is cancelled if we exit early
        if not interaction_task.done():
            interaction_task.cancel()



# ----------------------
# Server startup and configuration
# ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Universal AI Interaction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")
    parser.add_argument("--dev", action="store_true", help="Enable development mode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default model to use")
    
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.dev:
        os.environ["DEV_MODE"] = "true"
    
    os.environ["DEFAULT_MODEL"] = args.model
    
    # Run uvicorn server
    import uvicorn
    
    # When running as a module or script, use the current app instance directly
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.dev,
        log_level="info"
    )

if __name__ == "__main__":
    main()
