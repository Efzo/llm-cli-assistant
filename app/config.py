# app/config.py

import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found."
    )


MODEL = "gpt-5.6-luna"

MAX_OUTPUT_TOKENS = 500

TIMEOUT_SECONDS = 30.0

MAX_RETRIES = 2


SYSTEM_INSTRUCTIONS = """
You are a helpful AI assistant.
Answer questions clearly and concisely.
If you do not know something, say so.
"""