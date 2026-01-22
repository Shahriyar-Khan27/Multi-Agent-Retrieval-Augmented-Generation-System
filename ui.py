import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agentic_rag_assistant import run_agent
from utils import ingest_documents
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

@st.cache_resource
def setup_db():
    ingest_documents()

setup_db()

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Agent Assistant", layout="wide")
st.title("🤖 RAG Agent Assistant")
st.write("Ask me a question about the documents! I can summarize, reformat, and retrieve info.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_prompt = st.chat_input("Enter your query...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_agent(user_prompt)
                st.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                if "type" in response.keys():
                    st.write(f"Agent : {response['type']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")