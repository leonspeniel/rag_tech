import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.base.embeddings.base import similarity
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

#1. Loading
documents = SimpleDirectoryReader("../data_text").load_data()

#3. Storing

#Db init
db = chromadb.PersistentClient()
db_collection = db.get_or_create_collection("whitepapers")
db_vector_store = ChromaVectorStore(chroma_collection=db_collection)
chroma_storage_context = StorageContext.from_defaults(vector_store=db_vector_store)


#2 Indexing with chroma storage

vs_index = VectorStoreIndex.from_documents(documents, storage_context=chroma_storage_context)


#4. Querying

retriever = VectorIndexRetriever(index=vs_index, similarity_top_k=2)
response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer,
                                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

response = query_engine.query("what is saw.com?")
print(response)

