"""Tests for context-setting validation. Run: python -m unittest discover tests"""
import contextlib, io, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import chatbot

KEYS = ["MAX_CONTEXT_MESSAGES", "SUMMARIZE_AFTER", "KEEP_RECENT"]


class ConfigTests(unittest.TestCase):
    def settings(self, **env):
        old = {k: os.environ.get(k) for k in KEYS}
        try:
            for k in KEYS:
                os.environ.pop(k, None)
            os.environ.update(env)
            with contextlib.redirect_stderr(io.StringIO()):
                return chatbot._context_settings()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_defaults(self):
        self.assertEqual(self.settings(), (10, 16, 8))

    def test_valid_custom_values_are_kept(self):
        self.assertEqual(self.settings(MAX_CONTEXT_MESSAGES="20", SUMMARIZE_AFTER="30", KEEP_RECENT="10"), (20, 30, 10))

    def test_non_numbers_fall_back(self):
        self.assertEqual(self.settings(MAX_CONTEXT_MESSAGES="abc", KEEP_RECENT="", SUMMARIZE_AFTER="1.5"), (10, 16, 8))

    def test_zero_and_negative_fall_back(self):
        # KEEP_RECENT=0 used to slice the whole history as "recent" (messages[-0:])
        self.assertEqual(self.settings(KEEP_RECENT="0", MAX_CONTEXT_MESSAGES="-3"), (10, 16, 8))

    def test_keep_recent_cannot_exceed_send_window(self):
        self.assertEqual(self.settings(KEEP_RECENT="12", MAX_CONTEXT_MESSAGES="10"), (10, 16, 10))

    def test_summarize_after_must_exceed_keep_recent(self):
        self.assertEqual(self.settings(SUMMARIZE_AFTER="5"), (10, 10, 8))


if __name__ == "__main__":
    unittest.main()
