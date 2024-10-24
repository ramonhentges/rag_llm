# RAG com LLM

Este repositório contém exemplos de aplicações utilizando Recuperação de Informação Baseada em Conhecimento (RAG) com Modelos de Linguagem Grande (LLM). A versão do modelo usada é o LLaMA 3.2.

## Estrutura do Projeto

- **embedding.ipynb**: Este notebook é responsável por gerar e salvar embeddings, utilizando o Langchain e o sistema de indexação vetorial FAISS.
  
- **main.ipynb**: Notebook que executa uma aplicação de chat com base nos embeddings gerados. Aqui, você pode interagir com o modelo em tempo real, utilizando um fluxo de perguntas e respostas.

- **main.py**: Script que executa a aplicação de chat como uma interface web usando Streamlit. Ele permite que usuários finais façam consultas diretamente em uma interface amigável.

## Prompting no RAG

A técnica de prompting utilizada é o **Chain-of-Thought Prompting**, que é particularmente eficaz para elicitar raciocínio em LLMs. Esta abordagem consiste em fornecer ao modelo uma cadeia de raciocínio, o que melhora o desempenho em tarefas complexas como raciocínio aritmético, senso comum e manipulação simbólica.

Baseado em experimentos documentados, como o artigo *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"*.


