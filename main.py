from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="RAG/vector_store/chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type= "mmr",
    search_kwargs= {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5  # 0 -> More diverse, 1 -> Less diverse
    }
)

llm = ChatMistralAI(model= "mistral-small-2506")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """You are a helpful assistant that provides accurate and concise answers to questions based on the provided context.
    If the context does not contain the answer, respond with "I couldn't find the answer in the document" and do not make up an answer."""),
    ("human",
     """Context:
     {context}
     Question: 
     {question}
     """)
])

print("RAG system created")
print("Press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")