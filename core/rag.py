import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .utils import load_config

VECTOR_STORE_BASE_PATH = "data"
config = load_config() # Load config once

# --- Initialize Embeddings ---
try:
    if not config.get("azure_embedding_deployment"):
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME not set.")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=config["azure_embedding_deployment"],
        openai_api_version=config["azure_api_version"],
        azure_endpoint=config["azure_endpoint"],
        openai_api_key=config["azure_api_key"],
    )
except Exception as e:
    print(f"Error initializing Azure Embeddings: {e}")
    embeddings = None # Handle potential init failure

def get_vector_store_path(project_id: str) -> str:
    """Gets the path for a project's vector store."""
    project_id_safe = "".join(c if c.isalnum() else "_" for c in project_id) # Basic sanitization
    return os.path.join(VECTOR_STORE_BASE_PATH, project_id_safe)

def load_documents(file_paths: list[str]) -> list[Document]:
    """Loads documents from PDF and TXT files."""
    docs = []
    for file_path in file_paths:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif file_path.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())
            print(f"Successfully loaded {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Warning: Could not load file {os.path.basename(file_path)}. Error: {e}")
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=300,
        length_function=len
    )
    return text_splitter.split_documents(docs)

def create_or_update_vector_store(project_id: str, docs: list[Document]):
    """Creates a new vector store or updates an existing one for a project using FAISS."""
    if not embeddings:
        print("Error: Embeddings not initialized. Cannot create/update vector store.")
        return False
        
    store_path = get_vector_store_path(project_id)
    os.makedirs(store_path, exist_ok=True)

    if not docs:
        print(f"No processable documents provided for project {project_id}.")
        return False

    chunked_docs = chunk_documents(docs)
    if not chunked_docs:
        print(f"No text could be extracted or chunked for project {project_id}.")
        return False

    print(f"Processing {len(chunked_docs)} chunks for project {project_id}...")

    try:
        # --- FAISS Implementation ---
        if os.path.exists(os.path.join(store_path, "index.faiss")):
            print("Loading existing FAISS index...")
            # Be mindful of allow_dangerous_deserialization=True risk if index source is untrusted
            vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Adding {len(chunked_docs)} new chunks to existing index...")
            vector_store.add_documents(chunked_docs)
        else:
            print("Creating new FAISS index...")
            vector_store = FAISS.from_documents(chunked_docs, embeddings)

        vector_store.save_local(store_path)
        print(f"Vector store updated and saved for project {project_id} at {store_path}")
        return True
    except Exception as e:
        print(f"Error during vector store creation/update for {project_id}: {e}")
        return False


def get_retriever_for_project(project_id: str):
    """Loads the FAISS vector store for a project and returns a retriever."""
    if not embeddings:
        print("Error: Embeddings not initialized. Cannot get retriever.")
        return None

    store_path = get_vector_store_path(project_id)
    index_file = os.path.join(store_path, "index.faiss")

    if not os.path.exists(index_file):
        print(f"No vector store index found for project {project_id} at {index_file}")
        return None

    try:
        # Ensure embeddings instance is the same as used for creation/saving
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        # Increase 'k' to retrieve more chunks if needed, adjust based on context window and desired detail
        return vector_store.as_retriever(search_kwargs={'k': 4})
    except Exception as e:
        print(f"Error loading FAISS index for {project_id}: {e}")
        return None

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(f"--- Start Document Chunk ---\n{doc.page_content}\n--- End Document Chunk ---" for doc in docs)

def setup_rag_chain(llm, retriever):
    """Sets up the RAG chain using Langchain Expression Language (LCEL)."""
    template = """You are an assistant for question-answering tasks.
Use ONLY the following pieces of retrieved context to answer the question.
If you don't know the answer from the context provided, just say that you don't know. Don't try to make up an answer.
Keep the answer concise and relevant to the question.

Context:
{context}

Question: {question}

Answer:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain