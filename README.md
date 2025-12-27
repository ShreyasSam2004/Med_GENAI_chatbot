# Med_GENAI_chatbot

# 218085830711.dkr.ecr.us-east-1.amazonaws.com/medicalbot

# Medical Chatbot 🏥🤖

A RAG (Retrieval-Augmented Generation) based medical chatbot that answers medical questions using PDF documents as knowledge base. Built with LangChain, Flask, and powered by GPT-4o and Pinecone vector database.

## 🚀 Features

- **Intelligent Medical Q&A**: Answers medical questions based on uploaded PDF documents
- **RAG Architecture**: Combines document retrieval with GPT-4o for accurate responses
- **Vector Search**: Uses Pinecone for efficient semantic search
- **Web Interface**: Simple Flask-based chat interface
- **Concise Answers**: Provides brief, three-sentence maximum responses

## 🛠️ Tech Stack

- **Python** - Core programming language
- **LangChain** - Framework for building LLM applications
- **Flask** - Web framework for the chat interface
- **GPT-4o** - OpenAI's language model for generating responses
- **Pinecone** - Vector database for document embeddings
- **HuggingFace** - Sentence transformers for embeddings

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key
- Pinecone API Key

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd medical-chatbot
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

## 📁 Project Structure
```
medical-chatbot/
│
├── app.py                  # Main Flask application
├── src/
│   ├── helper.py          # Helper functions for data processing
│   └── prompt.py          # System prompt configuration
├── templates/
│   └── chat.html          # Chat interface template
├── data/                  # Directory for PDF documents
├── .env                   # Environment variables
└── requirements.txt       # Project dependencies
```

## 🎯 Usage

### 1. Prepare Your Data

Place your medical PDF documents in the `data/` directory.

### 2. Create Embeddings and Upload to Pinecone

Run the data processing script to extract text, create chunks, and upload to Pinecone:
```python
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
import os

# Load documents
extracted_data = load_pdf_file("data/")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(minimal_docs)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "medical-chatbot"

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Upload documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
```

### 3. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:8080`

### 4. Interact with the Chatbot

Open your browser and navigate to `http://localhost:8080`. Type your medical questions in the chat interface.

**Example Questions:**
- "What is Acromegaly and gigantism?"
- "What are the symptoms of diabetes?"
- "How is hypertension treated?"

## 🧩 Key Components

### Document Processing (`src/helper.py`)

- **load_pdf_file()**: Loads PDF documents from a directory
- **filter_to_minimal_docs()**: Filters document metadata to keep only source information
- **text_split()**: Splits documents into 500-character chunks with 20-character overlap
- **download_hugging_face_embeddings()**: Downloads the sentence-transformers model (384 dimensions)

### RAG Chain (`app.py`)

1. **Retriever**: Fetches top 3 most similar document chunks
2. **Prompt Template**: Formats the context and question for GPT-4o
3. **Question-Answer Chain**: Combines documents and generates answers
4. **RAG Chain**: Complete retrieval and generation pipeline

## ⚙️ Configuration

### Embedding Model
```python
model_name='sentence-transformers/all-MiniLM-L6-v2'  # 384 dimensions
```

### Chunk Settings
```python
chunk_size=500
chunk_overlap=20
```

### Retrieval Settings
```python
search_type="similarity"
search_kwargs={"k": 3}  # Retrieve top 3 documents
```

### LLM Model
```python
model="gpt-4o"
```

## 📦 Dependencies
```
flask
langchain
langchain-openai
langchain-pinecone
langchain-community
pinecone-client
python-dotenv
pypdf
sentence-transformers
```

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- Use environment variables for sensitive information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4o
- Pinecone for vector database
- HuggingFace for embedding models
- LangChain for the RAG framework

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice.
