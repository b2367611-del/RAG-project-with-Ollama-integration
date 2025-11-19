# RAG Project with Ollama Integration

A Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Ollama for querying PDF documents using local LLMs.

## Overview

This project implements a complete RAG pipeline that allows you to:
- Load and process PDF documents
- Create vector embeddings using Ollama
- Store embeddings in ChromaDB vector database
- Query documents using natural language
- Get AI-generated responses based on document context

## Features

- **Local LLM Integration**: Uses Ollama for both embeddings and text generation
- **Vector Database**: ChromaDB for efficient similarity search
- **Document Processing**: Automatic PDF loading and text chunking
- **Smart Retrieval**: Context-aware document retrieval
- **Testing Suite**: Includes tests to validate RAG performance

## Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/download) installed and running
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/b2367611-del/RAG-project-with-Ollama-integration.git
cd RAG-project-with-Ollama-integration
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama Models

Make sure Ollama is running, then pull the required models:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

## Usage

### 1. Prepare Your Documents

Place your PDF files in the `data/` directory:

```bash
mkdir data
# Copy your PDF files to the data/ directory
```

### 2. Populate the Database

Run the populate script to process PDFs and create the vector database:

```bash
python populate_database.py --reset
```

The `--reset` flag clears any existing database. Omit it to add documents to an existing database.

### 3. Query Your Documents

Ask questions about your documents:

```bash
python query_data.py "Your question here"
```

Example:

```bash
python query_data.py "What are the main topics discussed in the document?"
```

### 4. Run Tests

Execute the test suite to validate the RAG system:

```bash
pytest test_rag.py
```

## Project Structure

```
RAG-project-with-Ollama-integration/
├── data/                          # Directory for PDF documents
├── chroma/                        # Vector database storage (auto-generated)
├── get_embedding_function.py      # Embedding configuration
├── populate_database.py           # Database population script
├── query_data.py                  # Query interface
├── test_rag.py                    # Test suite
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## How It Works

1. **Document Loading**: PDFs from the `data/` directory are loaded using LangChain's PDF loader
2. **Text Splitting**: Documents are split into chunks (800 characters with 80 character overlap)
3. **Embedding**: Each chunk is converted to embeddings using Ollama's `nomic-embed-text` model
4. **Storage**: Embeddings are stored in ChromaDB with metadata
5. **Querying**: User questions are embedded and matched against stored chunks
6. **Generation**: Retrieved context is sent to Ollama's `mistral` model for answer generation

## Configuration

### Changing the Embedding Model

Edit `get_embedding_function.py`:

```python
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="your-model-name")
    return embeddings
```

### Changing the LLM Model

Edit `query_data.py` and `test_rag.py`:

```python
model = Ollama(model="your-model-name")
```

### Adjusting Chunk Size

Edit `populate_database.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Adjust this
    chunk_overlap=80,    # Adjust this
    length_function=len,
    is_separator_regex=False,
)
```

## Troubleshooting

### Ollama Connection Issues

Ensure Ollama is running:

```bash
# Check if Ollama is running
ollama list
```

### Import Errors

If you encounter import errors, ensure you're using Python 3.11 and have activated the virtual environment.

### Database Issues

To reset the database completely:

```bash
python populate_database.py --reset
```

## Dependencies

- `langchain` - LLM application framework
- `langchain-community` - Community integrations
- `langchain-core` - Core LangChain functionality
- `langchain-text-splitters` - Text splitting utilities
- `langchain-ollama` - Ollama integration
- `chromadb` - Vector database
- `pypdf` - PDF processing
- `pytest` - Testing framework
- `boto3` - AWS SDK (optional, for Bedrock embeddings)

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to submit issues or pull requests to improve the project!
