from dotenv import load_dotenv
import os

load_dotenv()  # reads the .env file

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "all-MiniLM-L6-v2"