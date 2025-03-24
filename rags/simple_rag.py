from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from common import helper
from common.helper import EmbeddingProvider

# Load environment variables from a .env file
load_dotenv()


path = "../data_pdf/Agents_whitepaper.pdf"

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = helper.replace_tab_with_space(texts)

    # Create embeddings (Tested with OpenAI and Amazon Bedrock)
    embeddings = helper.get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    #embeddings = helper.get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)

    # Create vector store
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

test_query = "What is an agent?"
context = helper.retrieve_context_per_question(test_query, chunks_query_retriever)
helper.show_context(context)


