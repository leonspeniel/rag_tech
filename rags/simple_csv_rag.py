import faiss

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from common.config import Config

# Initialize OpenAI language model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key= Config.OPENAI_API_KEY)

# Load CSV file
file_path = "../data_csv/user_data.csv"  # Insert the correct CSV file path

# Load and split documents from CSV
loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()

# Initialize FAISS vector store
embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vector_store.add_documents(documents=docs)

# Set up retriever
retriever = vector_store.as_retriever()

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Invoke the chain with a query
answer = rag_chain.invoke({"input": "Which company does Sheryl Baxter work for?"})
print(answer['answer'])