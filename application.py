import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("RAG Chatbot")
st.write("Upload documents and ask questions")

@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

embeddings = load_embeddings()

@st.cache_resource
def load_vectorstore():
    vectorstore = Chroma(
        persist_directory ="./chroma_db",
        embedding_function= embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 5}
)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_llm()

def load_document(file_path):
    if file_path.endswith(".txt"):
        return TextLoader(
            file_path,
            encoding="utf-8"
        ).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith(".csv"):
        return CSVLoader(file_path).load()
    elif file_path.endswith(".json"):
        return JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content = False
        ).load()
    elif file_path.endswith(".html"):
        return UnstructuredHTMLLoader(file_path).load()
    elif file_path.endswith(".md"):
        return UnstructuredMarkdownLoader(file_path).load()
    elif file_path.endswith(".pptx"):
        return UnstructuredPowerPointLoader(file_path).load()
    elif file_path.endswith(".xlsx"):
        return UnstructuredExcelLoader(file_path).load()
    return []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

def process_uploaded_files(uploaded_files):
    all_docs =[]

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.processed_files:
            st.warning(f"Skipped already processed file: {uploaded_file.name}")
            continue
        file_extension = os.path.splitext(uploaded_file.name)[1]

        status_text.text(f"Processing: {uploaded_file.name}")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

            try:
                docs = load_document(temp_path)
                for page_num, doc in enumerate(docs):
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["page"] = page_num + 1
                all_docs.extend(docs)

                st.success(f"Loaded: {uploaded_file.name}")
                st.session_state.processed_files.add(uploaded_file.name)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
            progress = (index + 1)/ total_files
            progress_bar.progress(progress)
    status_text.text("✅Processing completed!")
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=30
    )
    
    docs = splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i
    return docs

def add_to_vectorstore(docs):
    vectorstore.add_documents(docs)
    try:
        vectorstore.persist()
    except:
        pass

def generate_answer(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2,
        do_sample=False
    )
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return answer

def rag(query):
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return (
            "❌ Sorry, I cannot find relevant information in the document for your query.",
            []
        )
    
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
    You are a knowledgeable AI assistant.

    Answer the question ONLY from the provided context.

    Guidelines:
    - First give a clear conceptual definition
    - Ignore unrelated information
    - Do NOT generate information outside the context
    - Avoid copying large code blocks
    - If code exists, explain it briefly in words
    - Keep the answer concise and easy to understand
    - Maximum 5 lines
    - If the answer is not found in the context, respond exactly with:
    "Not found in document"
    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    answer = generate_answer(prompt)
    return answer, retrieved_docs

st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload Files",
    type=["txt",
          "pdf",
          "docx",
          "csv",
          "json",
          "html",
          "md",
          "pptx",
          "xlsx"
    ],
    accept_multiple_files=True
)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Process Files", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing Files...⏳"):
                documents = process_uploaded_files(
                    uploaded_files
                )
                split_docs = split_documents(
                    documents
                )
                add_to_vectorstore(split_docs)
            st.sidebar.success(
                "Documents processed successfully!"
            )
        else:
            st.sidebar.warning(
                "Please upload files"
            )

with col2:
    if st.button("Clear Database", use_container_width=True):
        try:
            vectorstore.delete_collection()
        except:
            pass
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        st.session_state.processed_files = set()
        st.sidebar.success("Database cleared successfully!")
query = st.text_input(
    "Enter your question",
    placeholder="Example: What is RAG?"
)

if st.button("Generate Answer") and query:
    with st.spinner("Generating Answer...⏳"):
        answer, docs = rag(query)
    st.markdown("## Answer")
    st.success(answer)
    if docs:
        st.markdown("## 📄 Sources")

        for i, doc in enumerate(docs):
            with st.expander(f"Source {i+1}"):
                st.write(doc.page_content)
                st.caption(
                    f"📁 File: {doc.metadata.get('source')} | "
                    f"📄 Page: {doc.metadata.get('page')}"
                )
    else: 
        st.warning(" No source found")