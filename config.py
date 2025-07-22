from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Validate API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("Missing GROQ_API_KEY in environment or .env file.")

# if not API_KEY:
#     raise EnvironmentError("Missing OPENAI_API_KEY in environment or .env file.")

def get_sqlite_checkpointer(db_path="data/chat_history.db"):
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(db_path)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)