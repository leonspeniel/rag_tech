from enum import Enum

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from common.config import config


def replace_tab_with_space(documents):

    for document in documents:
        document.page_content = document.page_content.replace('\t', " ")

    return documents


# Enum class representing different embedding providers
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"

# Enum class representing different model providers
class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"

def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Returns an embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str): Optional -  The specific embeddings model ID to use .

    Returns:
        A LangChain embedding provider instance.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    if provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key= config.OPENAI_API_KEY)
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def encode_pdf(path, chunk_size, chunk_overlap):

    loader = PyPDFLoader(path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(document)

    cleaned_texts = replace_tab_with_space(texts)

    embeddings = OpenAIEmbeddings(openai_api_key= config.OPENAI_API_KEY)
    vector_store = FAISS.from_documents(cleaned_texts, embeddings)

    return vector_store

def retrieve_context_per_question(question, chunk_query_retriever):

    documents = chunk_query_retriever.get_relevant_documents(question)

    context = [document.page_content for document in documents]

    return context

def show_context(context):

    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")