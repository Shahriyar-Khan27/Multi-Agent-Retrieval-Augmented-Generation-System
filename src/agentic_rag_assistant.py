import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    raise ValueError("LLAMA_API_KEY not found. Please set it in your .env file.")

# Local file paths and constants
DB_DIR = os.path.join(os.getcwd(), "chroma_db")
VECTOR_MODEL = "all-MiniLM-L6-v2"
LLAMA_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8" # Change if needed
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"

# Instantiate the components
embeddings = HuggingFaceEmbeddings(model_name=VECTOR_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- Llama API Call Helper ---
def llama_invoke(messages):
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages
    }
    response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Llama API error: {response.status_code} - {response.text}")

# --- Define the Agent's Tools ---

@tool
def retrieve_information(query: str) -> dict:
    """
    Retrieves relevant information from the document knowledge base for a given query.
    Returns the content and the source document names.
    """
    retriever = db.as_retriever()
    docs = retriever.invoke(query)
    source_metadata = ", ".join(list(set([doc.metadata.get("source") for doc in docs])))
    content = " ".join([doc.page_content for doc in docs])

    prompt = """You are RAG assistant. Based on the context : {context}, answer the user query
    to the user. Here's the user query : {user_query}. Be conscise and answer to the point dont be
    descriptive.""".format(context = content, user_query = query)

    llm_response = llm.invoke(prompt)

    return {"content": llm_response.content, "source": source_metadata, "type" : "rag"}

class SummarizeToolInput(BaseModel):
    content: str = Field(description="The text content to be summarized.")
    length: str = Field(description="Desired summary length, 'default' (100 words) or 'long'.")

def summarize_content(content: str, length: str = "default") -> dict:
    """
    Summarizes a block of text to a specified length using Llama API.
    """
    summary_prompt = f"You are a summarization expert.\nThe user wants a summary of the following text:\n\n\"{content}\"\n\n- If the requested length is 'default', provide a concise summary of about 100 words.\n- If the requested length is 'long', provide a detailed summary.\n\nSummary:"
    messages = [
        {"role": "system", "content": "You are a summarization expert."},
        {"role": "user", "content": summary_prompt}
    ]
    response = llm.invoke(messages)
    return {"output": response.content, "type" : "summarizer"}

class FormatToolInput(BaseModel):
    text: str = Field(description="The text to be reformatted.")
    format_type: str = Field(description="The desired format type: 'slack' or 'email'.")

def format_response(text: str, format_type: str) -> dict:
    """
    Reformats a given text for a specific context, either a Slack message or a formal email using Llama API.
    """

    
    format_prompt = f"You are a content formatter.\nThe user wants to reformat the following text:\n\n\"{text}\"\n\n- If the format is 'slack', format the text as a brief, bullet-pointed Slack message.\n- If the format is 'email', format the text as a professional email to an executive, starting with a subject line.\n\nFormatted text:"
    messages = [
        {"role": "system", "content": "You are a content formatter."},
        {"role": "user", "content": format_prompt}
    ]
    response = llm.invoke(messages)
    return {"output": response.content, "type" : "formatter"}

# --- Agent Creation ---
def process_query(query: str, chat_history=None) -> dict:
    """
    Process the query using dynamic dispatch instead of if/else.
    """
    query_lower = query.lower()
    intent = None
    length = "default"
    format_type = None

    # --- Intent Handlers ---
    def handle_summarize(query_lower):
        nonlocal length
        if "long" in query_lower:
            length = "long"
        summary_result = summarize_content(query_lower, length)
        return summary_result["output"], summary_result["type"], None

    def handle_rag(query):
        retrieval = retrieve_information.invoke(query)
        return retrieval["content"], "rag agent", retrieval["source"]

    def handle_format(query, fmt):
        format_result = format_response(query, fmt)
        return format_result["output"], format_result["type"], None

    # --- Dispatcher Map (replaces if/else) ---
    dispatcher = [
        ("summarize", lambda q: "summarize" in q, handle_summarize),
        ("rag", lambda q: "pdf" in q, handle_rag),
        ("slack", lambda q: "slack" in q, lambda q: handle_format(q, "slack")),
        ("email", lambda q: "email" in q, lambda q: handle_format(q, "email")),
    ]

    # --- Routing Logic ---
    output, response_type, source = None, None, None
    for intent_name, condition, handler in dispatcher:
        if condition(query_lower):
            intent = intent_name
            output, response_type, source = handler(query)
            break

    # --- Always include metadata ---
    if source:
        output += f"\n\n_Source: {source}_"

    if isinstance(output, dict) and "type" in output.keys():
        return {"output": output, "type": response_type}

    return {"output": output}


# --- Main function to run the agent ---
def run_agent(query: str):
    """
    Main function to run the agent with a query and print the result.
    """
    from src.utils import ingest_documents # Import here to avoid circular dependencies

    # Vector store setup
    # Loading the PDF, text spliting, and storing the embeddings of each split in the vector store (all-MiniLM-L6-v2)
    # Chunk size is 1000 charecters and overlap is 200. 
    
    ingest_documents() 
    try:
        result = process_query(query)
        return result
    except Exception as e:
        return {"output": f"An error occurred: {e}"}

if __name__ == "__main__":
    # Example usage for testing
    from src.utils import ingest_documents
    
    ingest_documents()
    
    # Load sample prompts
    prompts_path = os.path.join(os.getcwd(), "prompts.json")
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    for key, prompt in prompts.items():
        print(f"\n--- Running Prompt: {key} ---")
        response = run_agent(prompt)
        if isinstance(response, dict) and "output" in response:
            print(f"*Final Response*:\n{response['output']}\n")
        else:
            print(f"*Final Response*:\n{response}\n")