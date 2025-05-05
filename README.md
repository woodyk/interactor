# Interactor

A modular AI interaction framework with streaming, tool calling, context
history, session management, retry/backoff logic, multi-provider support,
and optional real-time logging.

Built for testability, extensibility, and CLI or programmatic use.

## Features

- Supports OpenAI, Ollama, NVIDIA, Google (Gemini) out-of-the-box
- Streamed output with markdown rendering
- Tool calling with auto-registration from Python functions
- Token-aware message cycling (to respect context limits)
- Persistent sessions with metadata and CRUD via `session.py`
- Optional per-instance logging of all interactions
- Retry logic for rate-limiting and transient failures
- CLI runner with interactive loop and safe shell command execution

## Installation

To install directly from GitHub:

    pip install git+https://github.com/woodyk/interactor.git#egg=interactor

Or clone locally and install for development:

    git clone https://github.com/woodyk/interactor.git
    cd interactor
    pip install -e .

## Requirements

Defined in `requirements.txt`:
- `openai`
- `aiohttp`
- `tiktoken`
- `rich`

## Usage (CLI)

After installing:

    interactor

Command-line options:

- `--model`: Model in format `provider:model` (e.g., `openai:gpt-4o`)
- `--api-key`: Optional override (default uses env var)
- `--base-url`: Optional custom API base
- `--tools`: Enable tool calling
- `--stream`: Stream assistant replies
- `--markdown`: Render streamed output as markdown

Use `/exit` or `/quit` to end the session.

## Usage (Python)

```python
from interactor import Interactor, Session

inter = Interactor(
    model="openai:gpt-4o",
    session_enabled=True,
    session_path="~/.interactor_sessions",
    log_path="~/interactor.log"
)

def say_hello(name: str) -> str:
    return f"Hello, {name}!"

inter.add_function(say_hello)

reply = inter.interact("Please greet Ada.")
print(reply)
```

## Session Support

Sessions are JSON files stored on disk.

You can:
- List and load sessions
- Create branches from message midpoints
- Search and update content
- Automatically summarize a session with `interactor.summarize()`

## Logging

To enable logging:

```python
inter = Interactor(..., log_path="~/my_session.log")
```

Logs include:
- User and assistant messages
- Tool call arguments and results
- Model switches, session transitions
- Retry events and exceptions

## License

Copyright (c) Wadih Khairallah. All rights reserved.

Licensed for use and modification under the MIT License.

