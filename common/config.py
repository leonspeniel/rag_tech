import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # LLMs keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")


config = Config()