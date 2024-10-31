@echo off
echo Executando Chat LLM.
E:
cd "E:\Inatel\Projetos\rag_llm"
call conda activate inatel
streamlit run main.py
