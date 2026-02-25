import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

DB_DIR = os.path.join(os.getcwd(), "chroma_db")
VECTOR_MODEL = "all-MiniLM-L6-v2"

# Instantiate embeddings (DB will be loaded when needed)
embeddings = HuggingFaceEmbeddings(model_name=VECTOR_MODEL)
db = None  # Will be initialized after document ingestion


def _get_db():
    """Initialize and return the ChromaDB instance."""
    global db
    if db is None:
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db


def _retrieve_docs(query: str, k: int = 5) -> tuple:
    """Retrieve relevant document chunks from ChromaDB. Returns (content, sources)."""
    vector_db = _get_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    sources = ", ".join(list(set([doc.metadata.get("source", "unknown") for doc in docs])))
    content = " ".join([doc.page_content for doc in docs])
    return content, sources


# --- Intent Classification using LLM ---
def classify_intent(query: str, chat_history: list = None) -> dict:
    """Use the LLM to classify user intent instead of brittle keyword matching."""
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 exchanges
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content[:200]}")
        history_context = "\nRecent conversation:\n" + "\n".join(history_lines)

    classification_prompt = f"""Classify the user's intent into exactly one category. Reply with ONLY the JSON object, no other text.

Categories:
- "greeting": casual greetings like hi, hello, hey, thanks, bye
- "summarize": user wants a summary of the documents (may say "summarize", "summary", "summarize the docs", "give me a summary", "key points", "overview", "brief")
- "rag": user wants to retrieve/ask about information from uploaded documents (asking about topics, wanting details, "tell about X", "what is X", "explain X", etc.)
- "format_slack": user wants content formatted as a Slack message
- "format_email": user wants content formatted as a professional email
- "conversation": general chat not related to documents
{history_context}

User query: "{query}"

Reply as JSON: {{"intent": "<category>", "length": "default"}}
For summarize, set length to "long" if user asks for detailed/long summary, otherwise "default"."""

    response = llm.invoke(classification_prompt)
    response_text = response.content.strip()

    # Parse the JSON response
    try:
        # Handle markdown code blocks
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        result = json.loads(response_text)
        return result
    except (json.JSONDecodeError, IndexError):
        # Fallback: extract intent from text
        text_lower = response_text.lower()
        if "greeting" in text_lower:
            return {"intent": "greeting"}
        elif "summarize" in text_lower:
            return {"intent": "summarize", "length": "default"}
        elif "rag" in text_lower:
            return {"intent": "rag"}
        elif "format_slack" in text_lower or "slack" in text_lower:
            return {"intent": "format_slack"}
        elif "format_email" in text_lower or "email" in text_lower:
            return {"intent": "format_email"}
        # Default to RAG for document-related app
        return {"intent": "rag"}


# --- Define the Agent's Tools ---

@tool
def retrieve_information(query: str) -> dict:
    """
    Retrieves relevant information from the document knowledge base for a given query.
    Returns the content and the source document names.
    """
    content, sources = _retrieve_docs(query, k=5)

    prompt = """You are an expert assistant. Use the provided context to answer the question accurately.

Context Information:
{context}

Question: {user_query}

Instructions:
- Answer using ONLY the information from the context
- Provide a direct, professional answer
- Do not mention "based on context" or "according to document"
- If the context contains multiple definitions or explanations, use the most relevant one
- Be accurate to the source material

Answer:""".format(context=content, user_query=query)

    llm_response = llm.invoke(prompt)

    return {"content": llm_response.content, "source": sources, "type": "rag"}


def summarize_content(content: str, length: str = "default") -> dict:
    """
    Summarizes a block of text to a specified length.
    """
    if length == "long":
        length_instruction = "Provide a detailed, comprehensive summary."
    else:
        length_instruction = "Provide a concise summary of about 100 words."

    summary_prompt = f"You are a summarization expert.\nSummarize the following text:\n\n\"{content}\"\n\n{length_instruction}\n\nSummary:"
    messages = [
        {"role": "system", "content": "You are a summarization expert."},
        {"role": "user", "content": summary_prompt}
    ]
    response = llm.invoke(messages)
    return {"output": response.content, "type": "summarizer"}


def format_response(text: str, format_type: str) -> dict:
    """
    Reformats a given text for a specific context, either a Slack message or a formal email.
    """
    format_prompt = f"You are a content formatter.\nThe user wants to reformat the following text:\n\n\"{text}\"\n\n- If the format is 'slack', format the text as a brief, bullet-pointed Slack message.\n- If the format is 'email', format the text as a professional email to an executive, starting with a subject line.\n\nFormatted text:"
    messages = [
        {"role": "system", "content": "You are a content formatter."},
        {"role": "user", "content": format_prompt}
    ]
    response = llm.invoke(messages)
    return {"output": response.content, "type": "formatter"}


# --- Intent Handlers ---

def handle_rag(query: str) -> tuple:
    """Retrieve from documents and answer the query."""
    retrieval = retrieve_information.invoke(query)
    return retrieval["content"], "rag", retrieval["source"]


def handle_summarize(query: str, length: str = "default") -> tuple:
    """Retrieve document content first, then summarize it."""
    # Retrieve relevant document content from ChromaDB
    doc_content, sources = _retrieve_docs(query, k=10)

    if not doc_content.strip():
        return "No documents found to summarize. Please upload PDFs first.", "summarizer", None

    summary_result = summarize_content(doc_content, length)
    return summary_result["output"], "summarizer", sources


def handle_format(query: str, format_type: str) -> tuple:
    """Retrieve document content first, then format it."""
    # Retrieve relevant content to format
    doc_content, sources = _retrieve_docs(query, k=5)

    if not doc_content.strip():
        format_result = format_response(query, format_type)
    else:
        format_result = format_response(doc_content, format_type)

    return format_result["output"], "formatter", sources


def handle_conversation(query: str, chat_history: list = None) -> tuple:
    """Handle general conversational queries using LLM directly."""
    history_context = ""
    if chat_history:
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content[:200]}")
        history_context = "\nConversation so far:\n" + "\n".join(history_lines) + "\n"

    conversation_prompt = f"You are a helpful AI assistant for a document Q&A system.{history_context}\nUser: {query}\n\nRespond naturally and concisely."
    response = llm.invoke(conversation_prompt)
    return response.content, "conversational", None


# --- Main Query Processor ---
def process_query(query: str, chat_history: list = None) -> dict:
    """
    Process the query using LLM-based intent classification.
    """
    # Classify intent using LLM
    intent_result = classify_intent(query, chat_history)
    intent = intent_result.get("intent", "rag")
    length = intent_result.get("length", "default")

    # Route to the appropriate handler
    if intent == "greeting":
        output, response_type, source = handle_conversation(query, chat_history)
    elif intent == "summarize":
        output, response_type, source = handle_summarize(query, length)
    elif intent == "format_slack":
        output, response_type, source = handle_format(query, "slack")
    elif intent == "format_email":
        output, response_type, source = handle_format(query, "email")
    elif intent == "rag":
        output, response_type, source = handle_rag(query)
    elif intent == "conversation":
        output, response_type, source = handle_conversation(query, chat_history)
    else:
        # Default to RAG for unknown intents
        output, response_type, source = handle_rag(query)

    # Append source metadata if available
    if source:
        output += f"\n\n_Source: {source}_"

    return {"output": output, "type": response_type}


# --- Main function to run the agent ---
def run_agent(query: str, chat_history: list = None):
    """
    Main function to run the agent with a query and return the result.
    """
    try:
        result = process_query(query, chat_history)
        return result
    except Exception as e:
        return {"output": f"An error occurred: {e}", "type": "error"}


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
