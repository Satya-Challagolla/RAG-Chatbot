import os
import re
import tempfile
import streamlit as st

from langchain_core.documents import Document
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
        collection_name= "rag_collection",
        persist_directory ="./chroma_db",
        embedding_function= embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5 }
)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
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

def build_pageindex(documents):

    indexed_docs = []

    for doc in documents:

        text = doc.page_content.strip()

        if not text:
            continue

        # Better logical splitting
        sections = re.split(
            r'\n\s*\n',
            text
        )

        for idx, section in enumerate(sections):

            section = section.strip()

            # Skip tiny sections
            if len(section) < 30:
                continue

            # First line becomes title
            lines = section.split("\n")

            title = lines[0][:80]

            new_doc = Document(
                page_content=section,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "section_id": idx,
                    "section_title": title
                }
            )

            indexed_docs.append(new_doc)

    return indexed_docs

#Create document tree
def create_document_tree(docs):
    tree ={}
    for doc in docs:
        source = doc.metadata["source"]
        if source not in tree:
            tree[source] =[]
        tree[source].append({
            "page": doc.metadata["page"],
            "section": doc.metadata["section_title"]
        })
    return tree

def add_to_vectorstore(docs):
    if not docs:
        st.error("No valid sections found.")
        return
    
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
        max_new_tokens=250,
        temperature=0.2,
        do_sample=False
    )
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    return answer

def rag(query):
    raw_docs = retriever.invoke(query)

    seen = set()
    retrieved_docs = []

    for doc in raw_docs:

        content = doc.page_content.strip()

        if content not in seen:
            seen.add(content)
            retrieved_docs.append(doc)
        
    context = "\n\n".join([
        f"""
        SECTION:
        {doc.metadata.get('section_title')}

        CONTENT:
        {doc.page_content}
        """
        for doc in retrieved_docs
    ])

    prompt = f"""
    You are a knowledgeable AI assistant.

    Answer the question using ALL relevant document sections.

    Instructions:
    - Combine information from multiple sections
    - Give a complete conceptual explanation
    - Mention important properties
    - Keep answer concise but informative
    - Do not hallucinate
    - If answer not found, say:
    "Not found in document"

    DOCUMENT SECTION:
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
                indexed_docs = build_pageindex(
                    documents
                )
                add_to_vectorstore(
                    indexed_docs
                )
                tree = create_document_tree(
                    indexed_docs
                )
                st.sidebar.subheader(
                    "📑 Document Structure"
                )
                st.sidebar.json(tree)
            
            st.sidebar.success(
                "✅ Documents processed successfully!"
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
            collection_name="rag_collection",
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":5}
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
        st.markdown("## 📄 Retrieved Sections")

        for i, doc in enumerate(docs):
            with st.expander(
                f"""
                Section {i+1}
                | {doc.metadata.get('section_title')}
                """
            ):
                st.write(doc.page_content)
                st.caption(
                    f"""
                    📁 File: {doc.metadata.get('source')}
                    📄 Page: {doc.metadata.get('page')}
                    🧩 Section ID: {doc.metadata.get('section_id')}
                    """
                )
    else: 
        st.warning(" No source found")