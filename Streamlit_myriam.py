from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from main import rag_document_embeddings
import openai
import warnings
from utils import prompt_output
from dotenv import load_dotenv
import streamlit as st
import yaml
import os

warnings.filterwarnings('ignore')

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RESULTS = 5

with st.sidebar:

    file_type = st.selectbox("Choose a file type", ['pdf', 'docx'])
    if file_type == 'pdf':
        uploaded_file = st.file_uploader("Choose a file", type=['pdf'], accept_multiple_files=False)
    elif file_type == 'xlsx':
        uploaded_file = st.text_input("Enter the Excel file path")
    elif file_type == "docx":
        uploaded_file = st.text_input("Enter the Docx file path")

if file_type == "pdf":
    st.title("PDF document Q&A")
if file_type == "docx":
    st.title("Word document Q&A")

llm_model = ChatOpenAI()

@st.cache_resource(show_spinner=True)
def load_qa(type_file):

    docsearch = rag_document_embeddings(uploaded_file,
                                        separator="\n",
                                        chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVERLAP,
                                        embedding_function=OpenAIEmbeddings(show_progress_bar=False),
                                        type_file=type_file
                                        )

    return docsearch


if uploaded_file not in [None, ""]:

    if (file_type=="pdf") or (file_type=="xlsx" and uploaded_file.endswith('.xlsx')) or (file_type=="docx" and uploaded_file.endswith('.docx')):

        if 'previous_file' not in st.session_state or st.session_state.previous_file != uploaded_file:
            st.session_state.previous_file = uploaded_file
            st.cache_resource.clear()
            st.session_state.messages = []

            if file_type == "pdf":
                docsearch = load_qa(type_file="pdf")
            elif file_type == "docx":
                try:
                    docsearch = load_qa(type_file="docx")
                except ValueError:
                    st.warning("File not found, check the file path.")

        else:

            if file_type == "pdf":
                docsearch = load_qa(type_file="pdf")
            elif file_type == "docx":
                try:
                    docsearch = load_qa(type_file="docx")
                except ValueError:
                    st.warning("File not found, check the file path.")

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])

        prompt = st.chat_input('Ask your questions here')

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({'role':'user', 'content':prompt})

            if file_type in ["pdf", "docx"]:
                response = prompt_output(prompt=prompt,
                                        embeddings=docsearch,
                                        model=llm_model,
                                        k_results=K_RESULTS
                                        )

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({"role":"assistant", "content":response})

    else:
        st.warning("Wrong file format !")