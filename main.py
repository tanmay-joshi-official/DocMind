import os
import shutil
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="wide"
)

embedding_model = HuggingFaceEmbeddings()

llm = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers only from the provided context.
If the answer is not present, reply:
"I couldn't find the answer in the document."
Do not make up information."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 Chat with your PDF")

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_pdf is not None:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join("uploads", uploaded_pdf.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    if os.path.exists("vector_store/chroma_db"):
        shutil.rmtree("vector_store/chroma_db")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="vector_store/chroma_db"
    )

    st.success("PDF processed successfully!")

    vectorstore = Chroma(
        persist_directory="vector_store/chroma_db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about the PDF")

    if question:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context":context,
                "question":question
            }
        )

        response = llm.invoke(final_prompt)

        answer = response.content

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )