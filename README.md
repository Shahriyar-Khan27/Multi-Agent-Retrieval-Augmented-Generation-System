# 🤖 Multi-Agent Retrieval-Augmented Generation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful **agent-based Retrieval-Augmented Generation (RAG) system** that dynamically retrieves, summarizes, and reformats information from documents using intelligent multi-agent orchestration.

## 🎯 Overview

This project demonstrates a sophisticated RAG system that leverages multiple AI agents to intelligently process and respond to user queries. The system can retrieve information from a knowledge base of PDF documents, summarize content, and reformat responses for different contexts (e.g., Slack messages, executive emails).

## 🏗️ Architecture

View the detailed system architecture: [Architecture Diagram](https://app.eraser.io/workspace/YivaLFUpXeQ1stQbSMLY)

![Architecture Overview](Documentation/Architecture%20RAG.png)

The system uses a multi-agent approach where:
- **Retrieval Agent**: Searches and retrieves relevant information from the vector database
- **Summarization Agent**: Condenses information to specified lengths
- **Formatting Agent**: Reformats content for different communication contexts
- **Orchestrator**: Coordinates agent activities using LangGraph

## 🚀 Features

- **🔍 Dynamic Tool Usage**: Intelligent agent automatically selects appropriate tools based on query context
- **📚 RAG Implementation**: Retrieves information from a knowledge base of PDF documents using vector embeddings
- **📝 Content Summarization**: Summarizes content to default or user-specified lengths
- **🎨 Flexible Formatting**: Reformats outputs for various contexts (Slack messages, emails, reports)
- **🔗 Source Transparency**: All responses include metadata about information sources
- **💬 Interactive UI**: User-friendly Streamlit interface for seamless interaction
- **⚡ Fast Inference**: Powered by Groq for rapid response generation

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq (Fast inference engine) |
| **Framework** | LangChain & LangGraph |
| **Vector Store** | ChromaDB (Local embeddings) |
| **UI** | Streamlit |
| **Language** | Python 3.8+ |
| **Document Processing** | PyPDF, LangChain Document Loaders |

## 📋 Prerequisites

Before running this project, ensure you have:

- Python 3.8 or higher installed
- pip (Python package manager)
- A Groq API key ([Get one here](https://console.groq.com/))
- PDF documents to use as your knowledge base

## 💻 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shahriyar-Khan27/Multi-Agent-Retrieval-Augmented-Generation-System.git
   cd Multi-Agent-Retrieval-Augmented-Generation-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Add your documents**

   Place your PDF files in the `documents/` folder

## 📂 Project Structure

```
Multi-Agent-RAG-System/
├── src/
│   ├── agentic_rag_assistant.py    # Core agent logic and orchestration
│   └── utils.py                     # Document ingestion and utilities
├── documents/                       # PDF knowledge base (add your files here)
├── Documentation/                   # Project documentation and diagrams
├── chroma_db/                       # Vector database (auto-generated)
├── ui.py                           # Streamlit web interface
├── prompts.json                    # Sample prompts for testing
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
└── README.md                      # This file
```

## 🚀 Usage

### Option 1: Command Line Interface

Test the core agent logic directly:

```bash
python src/agentic_rag_assistant.py
```

### Option 2: Web Interface (Recommended)

Launch the interactive Streamlit UI:

```bash
streamlit run ui.py
```

The application will:
1. Automatically ingest documents from the `documents/` folder
2. Build the vector database using ChromaDB
3. Launch the web interface (usually at `http://localhost:8501`)

### Example Queries

Try these sample queries:
- "What are the main findings in the research paper?"
- "Summarize the document in 3 sentences"
- "Reformat this as a Slack message"
- "What does the paper say about facial expression recognition?"

## ⚙️ Configuration

### Prompts Configuration

Modify `prompts.json` to add custom prompt templates:

```json
{
  "sample_queries": [
    "Your custom query here",
    "Another sample query"
  ]
}
```

### Agent Configuration

Edit agent parameters in `src/agentic_rag_assistant.py`:
- Model selection
- Temperature settings
- Max tokens
- Retrieval parameters

## 📚 Documentation

Detailed documentation is available in the `Documentation/` folder:
- [Full Documentation PDF](Documentation/Documentation%20-%20By%20SK.pdf)
- [Architecture Diagram](Documentation/Architecture%20RAG.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shahriyar Khan**

- GitHub: [@Shahriyar-Khan27](https://github.com/Shahriyar-Khan27)

## 🙏 Acknowledgments

- LangChain for the agent orchestration framework
- Groq for fast LLM inference
- ChromaDB for efficient vector storage
- Streamlit for the intuitive UI framework

---

**Note**: This is a prototype system designed for demonstration and learning purposes. For production use, consider additional security measures, error handling, and scalability improvements.
