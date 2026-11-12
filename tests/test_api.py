"""API-level tests (Flask test client + fake LLM client; no key or network needed)."""
import os, sys, unittest
from types import SimpleNamespace

import httpx
from openai import APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app as asnivo
import chatbot
from chatbot import ChatBot, ConversationStore

REQ = httpx.Request("POST", "http://x")


def resp(status):
    return httpx.Response(status, request=REQ)


def ok(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """Records every request. `behaviour` may be an exception to raise or a
    callable returning a response object."""

    def __init__(self, behaviour=None):
        self.calls, self.behaviour = [], behaviour
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw["messages"])
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        if callable(self.behaviour):
            return self.behaviour()
        if kw["messages"][0]["content"].startswith("Update the running summary"):
            return ok("SUMMARY: the user is called Asha")
        return ok("reply")


def texts(messages):
    return " | ".join(m["content"] for m in messages)


class ApiTests(unittest.TestCase):
    def setUp(self):
        chatbot.MAX_CONTEXT_MESSAGES, chatbot.SUMMARIZE_AFTER, chatbot.KEEP_RECENT = 6, 10, 4
        self.fake = FakeClient()
        asnivo.bot = ChatBot(self.fake)
        asnivo.store = ConversationStore()
        self.c = asnivo.app.test_client()

    def chat(self, message, cid=None):
        body = {"message": message}
        if cid:
            body["conversation_id"] = cid
        return self.c.post("/api/chat", json=body)

    # --- endpoint + memory ---------------------------------------------------
    def test_health_and_index(self):
        self.assertEqual(self.c.get("/api/health").get_json(), {"status": "ok"})
        page = self.c.get("/")
        self.assertIn(b"Asnivo", page.data)
        page.close()

    def test_memory_across_turns(self):
        cid = self.chat("my name is Asha").get_json()["conversation_id"]
        self.chat("I like Python", cid)
        r = self.chat("what is my name?", cid)
        self.assertEqual(r.status_code, 200)
        sent = self.fake.calls[-1]
        self.assertIn("my name is Asha", texts(sent))
        self.assertIn("I like Python", texts(sent))
        self.assertIn("Asnivo", sent[0]["content"])  # system prompt present

    def test_conversations_are_isolated(self):
        a = self.chat("alpha-secret").get_json()["conversation_id"]
        b = self.chat("beta-message").get_json()["conversation_id"]
        self.assertNotEqual(a, b)
        self.assertNotIn("alpha-secret", texts(self.fake.calls[-1]))
        self.chat("second beta", b)
        self.assertNotIn("alpha-secret", texts(self.fake.calls[-1]))
        self.chat("second alpha", a)
        self.assertNotIn("beta-message", texts(self.fake.calls[-1]))

    def test_reset_clears_history(self):
        cid = self.chat("remember-me").get_json()["conversation_id"]
        self.c.post("/api/reset", json={"conversation_id": cid})
        self.chat("hello again", cid)
        self.assertNotIn("remember-me", texts(self.fake.calls[-1]))

    # --- context management --------------------------------------------------
    def test_long_conversation_stays_bounded_and_keeps_summary(self):
        cid = self.chat("my name is Asha").get_json()["conversation_id"]
        for i in range(25):
            self.chat(f"filler {i}", cid)
        for call in self.fake.calls:
            self.assertLessEqual(len([m for m in call if m["role"] != "system"]), 6)
        last_chat_call = [c for c in self.fake.calls if c[0]["role"] == "system"][-1]
        self.assertEqual(len([m for m in last_chat_call if m["role"] == "system"]), 1)
        self.assertIn("Asha", last_chat_call[0]["content"])  # fact survived via summary
        self.assertNotIn("my name is Asha", texts(last_chat_call[1:]))  # original was dropped

    # --- validation ----------------------------------------------------------
    def test_bad_requests_return_json_400(self):
        cases = [
            {"message": ""}, {"message": "   "}, {"message": 123}, {"message": None},
            {"message": "x" * 2001}, {"message": "hi", "conversation_id": ["x"]},
        ]
        for body in cases:
            r = self.c.post("/api/chat", json=body)
            self.assertEqual(r.status_code, 400, body)
            self.assertIn("error", r.get_json())
        r = self.c.post("/api/chat", json=["hi"])
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/chat", data="not json", content_type="text/plain")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.fake.calls, [])  # nothing reached the LLM

    # --- error handling ------------------------------------------------------
    def test_api_failures_give_friendly_errors_and_keep_history_clean(self):
        cases = [
            (APITimeoutError(request=REQ), 504, "too long"),
            (RateLimitError("rl", response=resp(429), body=None), 429, "Rate limit"),
            (AuthenticationError("bad", response=resp(401), body=None), 500, "API key"),
            (APIStatusError("boom", response=resp(500), body=None), 502, "error (500)"),
            (RuntimeError("weird"), 500, "went wrong"),
        ]
        for exc, status, fragment in cases:
            asnivo.bot = ChatBot(FakeClient(exc))
            asnivo.store = ConversationStore()
            r = self.chat("hello")
            self.assertEqual(r.status_code, status, exc)
            self.assertIn(fragment, r.get_json()["error"])
            cid = r.get_json()["conversation_id"]
            _, conv = asnivo.store.get_or_create(cid)
            self.assertEqual(conv.messages, [])  # failed message rolled back

    def test_invalid_responses_are_handled(self):
        bad = [
            lambda: SimpleNamespace(choices=[]),
            lambda: SimpleNamespace(choices=None),
            lambda: ok(""),
            lambda: ok(None),
        ]
        for make in bad:
            asnivo.bot = ChatBot(FakeClient(make))
            r = self.chat("hello")
            self.assertEqual(r.status_code, 502, r.get_json())
            self.assertIn("error", r.get_json())

    def test_missing_api_key_message(self):
        os.environ.pop("LLM_API_KEY", None)
        asnivo.bot = ChatBot()  # real client path, no key
        r = self.chat("hi")
        self.assertEqual(r.status_code, 500)
        self.assertIn("LLM_API_KEY", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
