# Asnivo

**Intelligence that evolves with every conversation.**

*Asnivo: Artificial Systems for Nurturing Intelligence, Vision and Opportunities.*

Asnivo is an AI chatbot that demonstrates multi-turn conversation, per-conversation history and
context-window management. It acts as a patient coding mentor for beginners and early-career developers.

## Features
- Backend endpoint that sends messages to a real LLM API (OpenAI or Google Gemini) and returns the reply.
- Multi-turn memory kept separately for each conversation (`conversation_id`).
- Context-window management: truncation plus a running summary (see below).
- System prompt that defines Asnivo's role, style and boundaries.
- Clear user-facing errors for timeouts, rate limits, bad keys, connection failures and invalid responses.
- Web chat UI (with a New chat button) and a terminal interface.
- API key read from environment variables; `.env` is git-ignored.

## Technology stack
Python 3.9+, Flask, the official `openai` SDK, `python-dotenv`, and a plain HTML/CSS/JS frontend.

## Project structure
```
app.py            Flask routes: /api/chat, /api/reset, /api/health, serves the UI
chatbot.py        system prompt, conversation store, context management, error translation
cli.py            terminal interface
static/index.html chat UI
tests/            offline unit and API tests (fake LLM client, no key needed)
.env.example      environment variable template
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables
```bash
cp .env.example .env             # Windows: copy .env.example .env
```
Edit `.env` and set your key. Never commit `.env`.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_API_KEY` | API key (required) | - |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `LLM_BASE_URL` | Set only for Gemini (see `.env.example`) | OpenAI |
| `MAX_CONTEXT_MESSAGES` | Max messages sent to the model per request | 10 |
| `SUMMARIZE_AFTER` | Stored-history length that triggers summarising | 16 |
| `KEEP_RECENT` | Messages kept verbatim after summarising | 8 |
| `REQUEST_TIMEOUT_SECONDS` | LLM request timeout | 30 |

Context settings are validated at startup: invalid or inconsistent values (for example `KEEP_RECENT` larger than
`MAX_CONTEXT_MESSAGES`) fall back to safe values with a warning on the console.

Gemini is supported through its OpenAI-compatible endpoint: uncomment the Gemini block in `.env.example`.

## Running
```bash
python app.py        # web UI at http://localhost:5000
python cli.py        # terminal version (/reset to start over, /quit to exit)
```
Set `FLASK_DEBUG=1` only for local development.

## How context management works
1. Every message is stored in the conversation's history, and the system prompt is always sent first.
2. **Truncation:** each request sends only the last `MAX_CONTEXT_MESSAGES` messages, so the prompt stays bounded.
3. **Running summary:** when stored history grows beyond `SUMMARIZE_AFTER`, the older messages are
   summarised by the LLM. The summary is added to the system prompt and only the newest `KEEP_RECENT`
   messages are kept verbatim, so early facts (such as the user's name) survive truncation.
4. If summarising fails, the conversation keeps working with truncation alone and retries on the next turn.

## Tests
```bash
python -m unittest discover tests
```
The tests use a fake LLM client and cover memory across turns, isolation between conversations, truncation,
summarising, input validation and error handling. They do not call a real LLM.

## Limitations
Conversations live in server memory: they are lost on restart, and the oldest are evicted after 500.
