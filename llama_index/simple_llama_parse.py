from llama_cloud_services.parse import ResultType
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, set_global_handler
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse

from common.config import Config

set_global_handler("arize_phoenix", endpoint="https://llamatrace.com/v1/traces")

parser = LlamaParse(
    api_key=Config.LLAMA_PARSER_API_KEY,
    result_type=ResultType.MD,
    verbose=True
)

file_extractor = {".pdf":parser}
documents = SimpleDirectoryReader("../data_pdf", file_extractor=file_extractor).load_data()
vs_index = VectorStoreIndex.from_documents(documents)
query_engine = vs_index.as_query_engine(llm=OpenAI(model="gpt-4o-mini"))
response = query_engine.query("Does edev has anyone with aws skills and how much do they cost?")
print(response)
