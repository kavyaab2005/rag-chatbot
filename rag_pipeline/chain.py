import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ----------------------------
# FORMAT DOCUMENTS
# ----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------
# CUSTOM PROMPT
# ----------------------------
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say 'I don't have enough information.'

Context:
{context}"""),
    ("human", "{question}"),
])


# ----------------------------
# BUILD RAG CHAIN
# ----------------------------
def build_chain(persist_dir="./vectorstore"):

    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Load vectorstore
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # LLM
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,


    )

    # Chain — passes question as string to retriever
    chain = (
    {
        "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question": RunnableLambda(lambda x: x["question"]),
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

    return chain, retriever