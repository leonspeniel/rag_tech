from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.openai import OpenAI

from common.config import Config

openai_llm = OpenAI(model="gpt-4o-mini", api_key=Config.OPENAI_API_KEY)
documents_to_rag = SimpleDirectoryReader("../data_text").load_data()
vs_index = VectorStoreIndex.from_documents(documents_to_rag)
chat_engine = vs_index.as_chat_engine(chat_mode=ChatMode.BEST, llm=openai_llm)

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break

    print(f"Agent: {chat_engine.chat(user_input)}")