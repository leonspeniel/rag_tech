import os.path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, set_global_handler, StorageContext, \
    load_index_from_storage
from llama_index.llms.openai import OpenAI

from common.config import Config

set_global_handler("arize_phoenix", endpoint="https://llamatrace.com/v1/traces")

openai_llm = OpenAI(model="gpt-4o-mini", api_key=Config.OPENAI_API_KEY)

if os.path.exists("storage"):
    print("Loading from storage")
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    vs_index = load_index_from_storage(storage_context)

else:
    print("Loading from file")
    documents_to_rag = SimpleDirectoryReader("../data_pdf").load_data()
    vs_index = VectorStoreIndex.from_documents(documents_to_rag)
    vs_index.storage_context.persist(persist_dir="storage")

query_engine = vs_index.as_query_engine(llm=openai_llm)

response = query_engine.query("What is an agent and give 3 highlights?")

print(f"Answers: {response}")