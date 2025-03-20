from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

from common.config import Config

openai_llm = OpenAI(model="gpt-4o-mini", api_key=Config.OPENAI_API_KEY)
documents_to_rag = SimpleDirectoryReader("../data_pdf").load_data()
vs_index = VectorStoreIndex.from_documents(documents_to_rag)
query_engine = vs_index.as_query_engine(llm=openai_llm) # can do without llm too, but answers are plain

response = query_engine.query("What is an agent and give 3 highlights?")

print(f"Answers: {response}")