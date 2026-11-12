"""Offline tests with a fake LLM client (no API key or network needed).
Run: python -m unittest discover tests"""
import os, sys, unittest
from types import SimpleNamespace

import httpx
from openai import APIConnectionError

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import chatbot
from chatbot import ChatBot, ChatError, Conversation


class FakeClient:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        if self.fail:
            raise APIConnectionError(request=httpx.Request("POST", "http://x"))
        self.calls.append(kw["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class ContextTests(unittest.TestCase):
    def setUp(self):
        chatbot.MAX_CONTEXT_MESSAGES, chatbot.SUMMARIZE_AFTER, chatbot.KEEP_RECENT = 6, 10, 4

    def test_history_is_sent_with_system_prompt(self):
        fc, conv = FakeClient(), Conversation()
        bot = ChatBot(fc)
        bot.reply(conv, "my name is Asha")
        bot.reply(conv, "what is a list?")
        sent = fc.calls[-1]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("my name is Asha", [m["content"] for m in sent])

    def test_window_is_truncated_and_summary_prepended(self):
        fc, conv = FakeClient(), Conversation()
        bot = ChatBot(fc)
        for i in range(12):
            bot.reply(conv, f"question {i}")
        self.assertTrue(conv.summary)                       # summary was created
        self.assertLessEqual(len(conv.messages), 10)         # history was compacted
        last = fc.calls[-1]
        self.assertLessEqual(len([m for m in last if m["role"] != "system"]), 6)

    def test_failure_rolls_back_and_raises_friendly_error(self):
        conv = Conversation()
        bot = ChatBot(FakeClient(fail=True))
        with self.assertRaises(ChatError) as ctx:
            bot.reply(conv, "hello")
        self.assertEqual(conv.messages, [])
        self.assertIn("internet", ctx.exception.user_message)


if __name__ == "__main__":
    unittest.main()
