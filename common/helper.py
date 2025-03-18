
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from common.config import config


def replace_tab_with_space(documents):

    for document in documents:
        document.page_content = document.page_content.replace('\t', " ")

    return documents


def encode_pdf(path, chunk_size, chunk_overlap):

    loader = PyPDFLoader(path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(document)

    cleaned_texts = replace_tab_with_space(texts)

    embeddings = OpenAIEmbeddings(openai_api_key= config.OPENAI_API_KEY)
    vector_store = FAISS.from_documents(cleaned_texts, embeddings)

    return vector_store

def retriever_context_per_question(question, chunk_query_retriever):

    documents = chunk_query_retriever.get_relevant_documents(question)

    context = [document.page_content for document in documents]

    return context

def show_context(context):

    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")