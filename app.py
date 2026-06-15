import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_pipeline.ingest import load_documents, chunk_documents, build_vectorstore
from rag_pipeline.chain import build_chain

load_dotenv()

st.set_page_config(page_title="DocChat AI", page_icon="Brain", layout="wide")
st.title("DocChat - RAG Chatbot")
st.caption("Ask questions about your own documents using AI")

with st.sidebar:
    st.header("Upload Documents")
    st.caption("Supported: PDF, TXT")

    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base", use_container_width=True):
        if uploaded_files:
            with st.spinner("Reading and indexing documents..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    for f in uploaded_files:
                        path = os.path.join(tmpdir, f.name)
                        with open(path, "wb") as out:
                            out.write(f.read())
                    docs = load_documents(tmpdir)
                    chunks = chunk_documents(docs)
                    build_vectorstore(chunks)
            st.success("Done! Knowledge base built successfully!")
            st.session_state.chain, st.session_state.retriever = build_chain()
        else:
            st.warning("Please upload at least one file first!")

    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Upload your PDF or TXT files")
    st.markdown("2. Click Build Knowledge Base")
    st.markdown("3. Ask questions in the chat!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    with st.spinner("Loading existing knowledge base..."):
        try:
            st.session_state.chain, st.session_state.retriever = build_chain()
        except Exception as e:
            st.session_state.chain = None
            st.session_state.retriever = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask anything about your documents..."):
    if st.session_state.chain is None:
        st.error("Please upload documents and build the knowledge base first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain.invoke({
                        "question": query,
                        "chat_history": st.session_state.chat_history
                    })
                    st.markdown(result)

                    docs = st.session_state.retriever.invoke(query)
                    if docs:
                        with st.expander("Sources used"):
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.caption(doc.page_content[:300] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })
                    st.session_state.chat_history.append((query, result))

                except Exception as e:
                    st.error(f"Error: {str(e)}")