# 📄 DocMind : Intelligent PDF Question Answering using RAG

A Retrieval-Augmented Generation (RAG) application that lets users upload a PDF and ask natural language questions about its contents. The application retrieves the most relevant sections from the document using vector embeddings and generates accurate answers using the Mistral LLM.

Built with **Streamlit**, **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, and **Mistral AI**.

---

# 🎥 Demo

https://github.com/user-attachments/assets/236678ed-f71c-42ba-90d5-0dfb57432cd7

---

## ✨ Features

-  Upload any PDF document
-  Ask questions in natural language
-  Retrieval-Augmented Generation (RAG)
-  Semantic search using vector embeddings
-  Context-aware answers from the uploaded document
-  Fast document retrieval with ChromaDB
-  Powered by Mistral AI
-  Clean and interactive Streamlit interface
-  Displays retrieved context for transparency

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application |
| LangChain | RAG Pipeline |
| ChromaDB | Vector Database |
| HuggingFace Embeddings | Text Embeddings |
| Mistral AI | Large Language Model |
| PyPDFLoader | PDF Parsing |
| RecursiveCharacterTextSplitter | Document Chunking |
| dotenv | Environment Variable Management |

---

# 🏗️ Architecture

```
                +----------------+
                |   Upload PDF   |
                +-------+--------+
                        |
                        v
              +--------------------+
              |   PyPDFLoader      |
              +--------------------+
                        |
                        v
        +-------------------------------+
        | Recursive Character Splitter  |
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        | HuggingFace Embeddings         |
        +-------------------------------+
                        |
                        v
          +---------------------------+
          |     Chroma Vector DB      |
          +---------------------------+
                        |
        User Question   |
              |         |
              v         |
      +-----------------------+
      | Similarity Retrieval  |
      +-----------------------+
                |
                v
      +-----------------------+
      | Retrieved Context     |
      +-----------------------+
                |
                v
      +-----------------------+
      | Mistral LLM           |
      +-----------------------+
                |
                v
         Final Answer
```

---

# 📂 Project Structure

```
chat-with-pdf/
│
├── main.py
├── requirements.txt
├── .env
├── uploads/
├── vector_store/
├── README.md
└── assets/
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/chat-with-pdf-rag.git

cd chat-with-pdf-rag
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create a `.env` File

```env
MISTRAL_API_KEY=your_mistral_api_key
```

---

## 5. Run the Application

```bash
streamlit run main.py
```

---

# 📄 How It Works

### Step 1

Upload a PDF document.

---

### Step 2

The PDF is parsed using **PyPDFLoader**.

---

### Step 3

The extracted text is split into smaller chunks.

---

### Step 4

Each chunk is converted into vector embeddings using Hugging Face.

---

### Step 5

The embeddings are stored in ChromaDB.

---

### Step 6

When a user asks a question:

- The question is embedded.
- The most relevant chunks are retrieved.
- Retrieved context is passed to the Mistral LLM.
- The model generates an answer based only on the retrieved content.

---

# 🧠 RAG Pipeline

```
PDF
 │
 ▼
Load Text
 │
 ▼
Chunking
 │
 ▼
Embeddings
 │
 ▼
Vector Store
 │
 ▼
User Question
 │
 ▼
Similarity Search
 │
 ▼
Relevant Chunks
 │
 ▼
LLM
 │
 ▼
Answer
```

---

# 📦 Dependencies

```
streamlit

langchain

langchain-community

langchain-huggingface

langchain-mistralai

langchain-text-splitters

chromadb

sentence-transformers

pypdf

python-dotenv
```

---

# 📚 What I Learned

Through this project I gained hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Semantic Search
- Text Embeddings
- LangChain
- Prompt Engineering
- Document Chunking
- PDF Processing
- Streamlit Application Development
- LLM Integration
- Building AI-powered applications

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project, feel free to fork the repository, create a new branch, and submit a pull request.

---

# ⭐ Show Your Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

# 👨‍💻 Author

**Tanmay Joshi**

- GitHub: https://github.com/tanmay-joshi-official
- LinkedIn: https://linkedin.com/in/tanmay-joshi-
