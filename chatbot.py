"""Asnivo - core chatbot logic: system prompt, per-conversation history, context-window
management (truncation + running summary) and error handling.

Works with OpenAI or Google Gemini (through Gemini's OpenAI-compatible
endpoint) using the official `openai` SDK. Configure with environment variables.
"""
import os
import sys
import threading
import uuid
from collections import OrderedDict

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

# ---------------------------------------------------------------- persona ---
SYSTEM_PROMPT = """You are Asnivo, a patient and encouraging coding mentor for beginners \
and early-career developers.

How you behave:
- Explain concepts in plain language first, then show a short code example.
- Keep answers focused: usually under 200 words unless the user asks for depth.
- When the user shares broken code, point to the likely cause before giving the fix, \
so they learn to spot it themselves.
- Ask at most one clarifying question, and only when you truly cannot proceed.
- Remember what the user told you earlier in the conversation (their language, \
level, goals) and use it.

Boundaries:
- Stay on programming, tools and developer careers. For anything else, say so \
briefly and steer back to coding.
- If you are not sure about something, say so instead of guessing.
- Never ask for or repeat secrets such as API keys or passwords."""

# ----------------------------------------------------------------- config ---
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("LLM_BASE_URL") or None


def _warn(msg: str):
    print(f"[config] {msg}", file=sys.stderr)


def _setting(name, default, minimum, cast=int):
    """Read a numeric env var; fall back to the default if missing, not a
    number, or below `minimum`."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
    except ValueError:
        _warn(f"{name}={raw!r} is not a valid number; using {default}.")
        return default
    if value < minimum:
        _warn(f"{name}={value} is below the minimum {minimum}; using {default}.")
        return default
    return value


def _context_settings():
    """Return (max_context, summarize_after, keep_recent), kept consistent:
    keep_recent <= max_context (so kept messages are really sent) and
    summarize_after > keep_recent (so there is something to summarise)."""
    max_ctx = _setting("MAX_CONTEXT_MESSAGES", 10, 2)
    keep = _setting("KEEP_RECENT", 8, 2)
    after = _setting("SUMMARIZE_AFTER", 16, 3)
    if keep > max_ctx:
        _warn(f"KEEP_RECENT={keep} exceeds MAX_CONTEXT_MESSAGES={max_ctx}; using {max_ctx}.")
        keep = max_ctx
    if after <= keep:
        _warn(f"SUMMARIZE_AFTER={after} must be greater than KEEP_RECENT={keep}; using {keep + 2}.")
        after = keep + 2
    return max_ctx, after, keep


TIMEOUT = _setting("REQUEST_TIMEOUT_SECONDS", 30.0, 1, float)
MAX_CONTEXT_MESSAGES, SUMMARIZE_AFTER, KEEP_RECENT = _context_settings()
HARD_HISTORY_CAP = 60  # safety net if summarising keeps failing


class ChatError(Exception):
    """An error with a message that is safe to show to the end user."""

    def __init__(self, user_message: str, status: int = 502):
        super().__init__(user_message)
        self.user_message = user_message
        self.status = status


class Conversation:
    def __init__(self):
        self.messages = []  # [{"role": "user"|"assistant", "content": str}]
        self.summary = ""  # running summary of messages that were folded away
        self.lock = threading.Lock()


class ConversationStore:
    """In-memory store keyed by conversation id (oldest evicted past max_size)."""

    def __init__(self, max_size: int = 500):
        self._items = OrderedDict()
        self._max = max_size
        self._lock = threading.Lock()

    def get_or_create(self, cid=None):
        with self._lock:
            if isinstance(cid, str) and cid in self._items:
                self._items.move_to_end(cid)
                return cid, self._items[cid]
            cid = uuid.uuid4().hex
            self._items[cid] = Conversation()
            while len(self._items) > self._max:
                self._items.popitem(last=False)
            return cid, self._items[cid]

    def delete(self, cid):
        with self._lock:
            self._items.pop(cid, None)


class ChatBot:
    def __init__(self, client=None):
        self._client = client

    # -- API client (created lazily so the app can start without a key) ------
    @property
    def client(self):
        if self._client is None:
            key = os.getenv("LLM_API_KEY")
            if not key:
                raise ChatError(
                    "The server has no API key configured. Set LLM_API_KEY in .env.",
                    status=500,
                )
            self._client = OpenAI(api_key=key, base_url=BASE_URL, max_retries=1)
        return self._client

    # -- Context management ----------------------------------------------------
    def build_payload(self, conv: Conversation):
        """What we actually send: system prompt + running summary + last N messages."""
        window = conv.messages[-MAX_CONTEXT_MESSAGES:]
        while window and window[0]["role"] != "user":  # never start mid-exchange
            window = window[1:]
        system = SYSTEM_PROMPT
        if conv.summary:
            # One system message (some providers reject several)
            system += (
                "\n\nSummary of the earlier part of this conversation "
                "(use it for context, do not recite it):\n" + conv.summary
            )
        return [{"role": "system", "content": system}] + window

    def _compact(self, conv: Conversation):
        """Fold old messages into the running summary once history gets long."""
        if len(conv.messages) <= SUMMARIZE_AFTER:
            return
        old, recent = conv.messages[:-KEEP_RECENT], conv.messages[-KEEP_RECENT:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        prompt = (
            "Update the running summary of a conversation between a user and a "
            "coding mentor. Keep facts about the user (name, language, level, "
            "goals), decisions made, and unresolved questions. Max 150 words. "
            "Output only the summary.\n\n"
            f"Existing summary:\n{conv.summary or '(none)'}\n\n"
            f"New messages to fold in:\n{transcript}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=TIMEOUT,
                max_tokens=300,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if summary:
                conv.summary = summary
                conv.messages = recent
        except Exception:
            # Summarising is best-effort. The send window still truncates, so
            # the conversation keeps working; we retry on the next turn.
            if len(conv.messages) > HARD_HISTORY_CAP:
                conv.messages = recent

    # -- Main entry point --------------------------------------------------------
    def reply(self, conv: Conversation, user_text: str) -> str:
        with conv.lock:
            conv.messages.append({"role": "user", "content": user_text})
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL,
                    messages=self.build_payload(conv),
                    timeout=TIMEOUT,
                    max_tokens=700,
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ChatError("The model returned an empty reply. Please try again.")
            except ChatError:
                conv.messages.pop()  # keep history consistent on failure
                raise
            except Exception as exc:
                conv.messages.pop()
                raise self._translate(exc) from exc

            conv.messages.append({"role": "assistant", "content": text})
            self._compact(conv)
            return text

    @staticmethod
    def _translate(exc: Exception) -> ChatError:
        if isinstance(exc, APITimeoutError):
            return ChatError("The AI took too long to respond. Please try again.", 504)
        if isinstance(exc, APIConnectionError):
            return ChatError("Could not reach the AI service. Check your internet connection.", 503)
        if isinstance(exc, RateLimitError):
            return ChatError("Rate limit or quota reached. Wait a moment and try again.", 429)
        if isinstance(exc, AuthenticationError):
            return ChatError("The API key was rejected. Check LLM_API_KEY in .env.", 500)
        if isinstance(exc, APIStatusError):
            return ChatError(f"The AI service returned an error ({exc.status_code}). Please try again.", 502)
        if isinstance(exc, (IndexError, AttributeError, TypeError)):
            return ChatError("The AI service sent an unexpected response. Please try again.", 502)
        return ChatError("Something went wrong on the server. Please try again.", 500)
