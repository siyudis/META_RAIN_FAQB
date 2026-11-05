import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Company / institute branding (used in the system prompt)
INSTITUTE_NAME = os.getenv("INSTITUTE_NAME", "LearnUp Academy")
# Website/support info for fallback answers
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@learnup.ai")
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "1800-123-4567")
