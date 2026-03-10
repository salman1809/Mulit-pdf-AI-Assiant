import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Get API key from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Multi PDF Research Assistant", page_icon="📚")

st.title("📚 Multi-PDF Research Assistant (GenAI RAG)")
st.write("Upload multiple PDFs and ask questions about them.")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    documents = []

    with st.spinner("Loading PDFs..."):
        for uploaded_file in uploaded_files:

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            documents.extend(docs)

    st.success("PDFs loaded successfully!")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector database
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever()

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

    # RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    st.divider()

    question = st.text_input("Ask a question about the PDFs")

    if question:

        with st.spinner("Searching documents..."):
            response = qa_chain.invoke({"query": question})

        st.subheader("Answer")
        st.write(response["result"])
