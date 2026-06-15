import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load API key from .env file
load_dotenv()

# ----------------------------
# STEP 1: LOAD DOCUMENTS
# ----------------------------
def load_documents(data_dir="./data"):
    documents = []

    # Load TXT files
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents.extend(txt_loader.load())

    # Load PDF files
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents.extend(pdf_loader.load())

    print(f"Total documents loaded: {len(documents)}")
    return documents


# ----------------------------
# STEP 2: SPLIT INTO CHUNKS
# ----------------------------
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# ----------------------------
# STEP 3: SAVE TO VECTOR STORE
# ----------------------------
def build_vectorstore(chunks, persist_dir="./vectorstore"):
    print("Creating embeddings and saving to ChromaDB...")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print("Vector store saved successfully!")
    return vectorstore


# ----------------------------
# RUN IT
# ----------------------------
if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)