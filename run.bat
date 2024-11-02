@echo off
echo Executando Chat LLM.
E:
cd "E:\Projetos\rag_llm"
call conda activate LLM
streamlit run main.py
