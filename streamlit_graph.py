import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredExcelLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from EA.Config.configuration import (main_key, NEO4J_URI, NEO4J_PASSWORD, NEO4J_USERNAME, OPENAI_API_VERSION)

# Load environment variables from .env file
load_dotenv()

st.title("Document to Neo4j Knowledge Graph")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "xlsx"])

if uploaded_file is not None:
    start_time = time.time()  # Start the timer

    file_name = uploaded_file.name

    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if '.pdf' in file_name or '.PDF' in file_name:
        loader = PyPDFLoader(file_name)

    elif '.docx' in file_name or '.doc' in file_name:
        loader = Docx2txtLoader(file_name)

    elif '.xlsx' in file_name or '.xls' in file_name:
        loader = UnstructuredExcelLoader(file_name)

    documents = loader.load_and_split()

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=8000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)

    st.write("### Extracted Text Chunks")
    st.write(texts)

    # Initialize the Azure OpenAI LLM
    llm = AzureChatOpenAI(
        deployment_name="gpt-4-0125-preview",
    )

    llm_transformer = LLMGraphTransformer(llm=llm)

    # Convert documents to graph format
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    st.write("### Graph Documents")
    st.write(graph_documents)

    # Store Knowledge Graph in Neo4j
    graph_store = Neo4jGraph()
    graph_store.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    graph_store.refresh_schema()

    st.write("### Neo4j Graph Schema")
    st.write(graph_store.schema)

    end_time = time.time()  # End the timer
    time_taken = end_time - start_time

    st.success(f"Document processed and stored in Neo4j Knowledge Graph successfully! Time taken: {time_taken:.2f} seconds")
