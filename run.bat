@echo off
echo Executando Chat LLM.
D:
cd "D:\Projetos\rag_llm"
call conda activate LLM
streamlit run main.py
