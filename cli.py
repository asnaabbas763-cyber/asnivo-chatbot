"""Terminal version of Asnivo. Run: python cli.py"""
from chatbot import ChatBot, ChatError, Conversation

bot = ChatBot()
conv = Conversation()
print("Asnivo - Intelligence that evolves with every conversation. Type /reset to start over, /quit to exit.\n")

while True:
    try:
        text = input("you > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        continue
    if text == "/quit":
        break
    if text == "/reset":
        conv = Conversation()
        print("(conversation cleared)\n")
        continue
    try:
        print(f"\nasnivo > {bot.reply(conv, text)}\n")
    except ChatError as err:
        print(f"\n[error] {err.user_message}\n")
