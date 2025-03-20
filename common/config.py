import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # LLMs keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LLAMA_PARSER_API_KEY: str = os.getenv("LLAMA_PARSER_API_KEY")
    OTEL_EXPORTER_OTLP_HEADERS: str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")


config = Config()