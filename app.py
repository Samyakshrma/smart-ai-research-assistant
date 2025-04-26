import streamlit as st
import os
import time
from core.rag import (
    create_or_update_vector_store,
    get_retriever_for_project,
    load_documents,
    setup_rag_chain,
)
from core.utils import load_config
from langchain_openai import AzureChatOpenAI

# --- Page Config ---
st.set_page_config(page_title="RAG Assistant (Azure)", layout="wide")
st.title("📄 Multi-Project RAG Assistant (Azure OpenAI)")
st.write("Upload documents to a project and ask questions based on their content.")

# --- Load Config and Initialize LLM ---
config = load_config()
llm = None
if all([config.get("azure_endpoint"), config.get("azure_api_key"), config.get("azure_api_version"), config.get("azure_chat_deployment")]):
    try:
        llm = AzureChatOpenAI(
            openai_api_version=config["azure_api_version"],
            azure_endpoint=config["azure_endpoint"],
            azure_deployment=config["azure_chat_deployment"],
            openai_api_key=config["azure_api_key"],
            temperature=0.3, # Low temp for factual Q&A
            max_retries=2,
        )
        st.sidebar.success("Azure LLM Initialized.")
    except Exception as e:
        st.sidebar.error(f"LLM Init Error: {e}")
        st.error(f"Could not initialize the Azure Chat LLM. Please check configuration and Azure deployment status. Error: {e}")
else:
    st.sidebar.error("Azure Chat LLM configuration missing in .env")
    st.error("Azure Chat LLM configuration missing. Cannot proceed with Q&A.")


# --- Constants ---
VECTOR_STORE_BASE_PATH = "data"
UPLOAD_FOLDER = "uploads"
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Session State Initialization ---
if "current_project" not in st.session_state:
    st.session_state.current_project = "default_project"
if "messages" not in st.session_state:
    st.session_state.messages = {} # Dictionary to store messages per project
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False


# --- Project Management Sidebar ---
st.sidebar.header("Project Management")
#project_name_input = st.sidebar.text_input("Enter Project Name", st.session_state.current_project)
st.sidebar.header("Project Management")

# List all existing projects (folders inside /data)
existing_projects = [d for d in os.listdir(VECTOR_STORE_BASE_PATH) if os.path.isdir(os.path.join(VECTOR_STORE_BASE_PATH, d))]

# Let user select from existing or enter a new one
selected_project = st.sidebar.selectbox("Select Existing Project", existing_projects, index=existing_projects.index(st.session_state.current_project) if st.session_state.current_project in existing_projects else 0) if existing_projects else None
new_project_name = st.sidebar.text_input("Or Enter New Project Name", value=st.session_state.current_project)

# Determine final project name to use
project_name_input = new_project_name if new_project_name != selected_project else selected_project or new_project_name


if project_name_input != st.session_state.current_project:
    st.session_state.current_project = project_name_input
    st.session_state.rag_chain = None # Reset chain on project change
    st.session_state.retriever_ready = False
    # Clear messages for the new project if switching
    if st.session_state.current_project not in st.session_state.messages:
         st.session_state.messages[st.session_state.current_project] = []
    st.rerun() # Rerun to update UI elements linked to project name

project_name = st.session_state.current_project
st.sidebar.write(f"**Current Project:** `{project_name}`")

# Initialize message list for the current project if it doesn't exist
if project_name not in st.session_state.messages:
    st.session_state.messages[project_name] = []


# --- File Upload Sidebar ---
st.sidebar.subheader("Upload Documents")
if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = {}

# Initialize project-specific processed flag
if project_name not in st.session_state.uploaded_files_processed:
    st.session_state.uploaded_files_processed[project_name] = False

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files to this project",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=f"uploader_{project_name}"  # Unique key per project helps reset uploader
)

if uploaded_files and not st.session_state.uploaded_files_processed[project_name]:
    file_paths = []
    all_processed_successfully = True

    with st.spinner(f"Processing {len(uploaded_files)} file(s) for '{project_name}'..."):
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(UPLOAD_FOLDER, f"{project_name}_{uploaded_file.name}")
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(temp_path)
            except Exception as e:
                st.sidebar.error(f"Error saving {uploaded_file.name}: {e}")
                all_processed_successfully = False

        if file_paths:
            try:
                loaded_docs = load_documents(file_paths)
                if loaded_docs:
                    success = create_or_update_vector_store(project_name, loaded_docs)
                    if success:
                        st.sidebar.success(f"Processed {len(file_paths)} file(s). Project '{project_name}' updated.")
                        st.session_state.rag_chain = None  # Force reload of retriever/chain
                        st.session_state.retriever_ready = False
                        st.session_state.uploaded_files_processed[project_name] = True
                        st.rerun()
                    else:
                        st.sidebar.error("Vector store update failed. Check logs.")
                        all_processed_successfully = False
                else:
                    st.sidebar.warning("No text could be extracted from the uploaded files.")
                    all_processed_successfully = False
            except Exception as e:
                st.sidebar.error(f"Error during processing: {e}")
                all_processed_successfully = False
            finally:
                for p in file_paths:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception as e_rem:
                            print(f"Warning: could not remove temp file {p}. Error: {e_rem}")

elif uploaded_files and st.session_state.uploaded_files_processed[project_name]:
    st.sidebar.info("These files were already processed.")

# --- Setup RAG Chain for Current Project ---
if not st.session_state.rag_chain and llm:
    retriever = get_retriever_for_project(project_name)
    if retriever:
        st.session_state.rag_chain = setup_rag_chain(llm, retriever)
        st.session_state.retriever_ready = True
        st.sidebar.info(f"Ready to answer questions for project '{project_name}'.")
    else:
        # No retriever means no data or error loading
        st.sidebar.warning(f"No document data found or loaded for '{project_name}'. Upload files.")
        st.session_state.retriever_ready = False
elif not llm:
     st.sidebar.error("LLM not available. Cannot setup Q&A chain.")
     st.session_state.retriever_ready = False


# --- Chat Interface ---
st.subheader(f"Ask questions about documents in Project: `{project_name}`")

# Display chat messages
for message in st.session_state.messages[project_name]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(f"Ask about '{project_name}' docs..."):
    # Add user message to project's chat history
    st.session_state.messages[project_name].append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if not llm:
             full_response = "Error: LLM is not initialized. Cannot generate response."
             st.error(full_response)
        elif not st.session_state.retriever_ready or not st.session_state.rag_chain:
             full_response = f"Error: Document retriever for project '{project_name}' is not ready. Please upload documents and wait for processing."
             st.warning(full_response)
        else:
            with st.spinner("Thinking based on documents..."):
                try:
                    # Invoke the RAG chain
                    response = st.session_state.rag_chain.invoke(prompt)
                    full_response = response
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"An error occurred: {e}"
                    st.error(full_response) # Show error in UI

        # Add assistant response to project's chat history
        st.session_state.messages[project_name].append({"role": "assistant", "content": full_response})