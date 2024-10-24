import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

import faiss
import tempfile
import os
import time

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

output_folder = 'db_faiss_chat'

os.makedirs(output_folder, exist_ok=True)

st.set_page_config(
    page_title="📚 DocumentAI Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        border: none;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .chat-message {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .ai-message {
        background-color: #F3E5F5;
    }
    .stAlert {
        background-color: #fff3e0;
        border: none;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal com animação
st.markdown("""
    <h1 style='text-align: center; color: #2E7D32; animation: fadeIn 2s;'>
        🤖 DocumentAI Chat - Llama 3
    </h1>
    <p style='text-align: center; color: #666;'>
        Seu assistente inteligente para análise de documentos
    </p>
    """, unsafe_allow_html=True)

model_class = "ollama" 

def model_openai(model="gpt-4o-mini", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    return llm

def model_ollama(model="llama3.2", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

# Indexação e recuperação
def config_retriever(uploads, k, fetch_k):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(f'vectorstore/{output_folder}')

    retriever = vectorstore.as_retriever(search_type="mmr", 
                                         search_kwargs={'k': k, 'fetch_k': fetch_k})
    return retriever

# Carregar embeddings locais
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.load_local(f'vectorstore/{output_folder}', embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 8})
    return retriever

# Configuração da chain
def config_rag_chain(model_class, retriever):
    if model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    context_q_system_prompt = """
    Dado o histórico de chat e a pergunta de acompanhamento, que pode fazer referência ao contexto no histórico do chat, 
    formule uma pergunta independente que possa ser entendida sem o histórico do chat. 
    NÃO responda à pergunta, apenas reformule-a se necessário e, caso contrário, retorne-a como está.
    """
    context_q_user_prompt = "Question: {input}"
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", context_q_user_prompt),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=context_q_prompt)

    qa_prompt_template = """Você é um assistente virtual especializado em analisar e responder perguntas sobre documentos.
    Use os seguintes trechos de contexto recuperado para responder à pergunta.
    Se você não souber a resposta, diga honestamente que não sabe. Mantenha a resposta concisa e focada.
    Se for solicitado a listar as referências dos artigos, informações específicas do documento, faça-o de forma estruturada e clara.
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""

    qa_prompt = PromptTemplate.from_template(qa_prompt_template)

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

# Painel lateral com opção de escolha
st.sidebar.title("Configurações")
mode = st.sidebar.radio(
    "Escolha uma opção",
    ('Ler Embeddings Existentes', 'Gerar Novos Embeddings')
)

if mode == 'Gerar Novos Embeddings':
    uploads = st.sidebar.file_uploader(
        label="Enviar arquivos", type=["pdf"], accept_multiple_files=True
    )
    if not uploads:
        st.info("Por favor, envie algum arquivo para continuar")
        st.stop()

    k = 3
    fetch_k = 8
    if "retriever" not in st.session_state:
        st.session_state.retriever = config_retriever(uploads, k, fetch_k)

else:
    if "retriever" not in st.session_state:
        st.session_state.retriever = load_embeddings()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu chat LLM! Como posso ajudar você?"),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        rag_chain = config_rag_chain(model_class, st.session_state.retriever)

        result = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})

        resp = result['answer']
        st.write(resp)

        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'Página não especificada')

            ref = f":link: Fonte {idx}: *{file} - p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))
