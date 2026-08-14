import os
import uuid
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
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not present, reply exactly:

I couldn't find the answer in the document.
"""
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

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

st.title("📄 Chat with your PDF")

uploaded_pdf = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if (
    uploaded_pdf is not None
    and uploaded_pdf.name != st.session_state.processed_file
):

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_pdf.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        cleaned_chunks = []

        for chunk in chunks:

            text = chunk.page_content

            if not isinstance(text, str):
                text = str(text)

            text = "".join(
                c for c in text
                if not (0xD800 <= ord(c) <= 0xDFFF)
            )

            text = text.encode(
                "utf-8",
                "ignore"
            ).decode("utf-8")

            if text.strip():
                chunk.page_content = text
                cleaned_chunks.append(chunk)

        persist_dir = os.path.join(
            "vector_store",
            uuid.uuid4().hex
        )

        vectorstore = Chroma.from_documents(
            documents=cleaned_chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )

        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        st.session_state.processed_file = uploaded_pdf.name
        st.session_state.messages = []

    st.success("PDF processed successfully!")

# ---------------- CHAT SECTION ---------------- #

if st.session_state.retriever is not None:

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the PDF")

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                docs = st.session_state.retriever.invoke(question)

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": question
                })

                response = llm.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("Upload a PDF to start chatting.")