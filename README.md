# RAG com LLM

Este repositório contém exemplos de aplicações utilizando Recuperação de Informação Baseada em Conhecimento (RAG) com Modelos de Linguagem Grande (LLM). A versão do modelo usada é o LLaMA 3.2.

## Estrutura do Projeto

- **jupyter.ipynb**: Notebook que executa uma aplicação de chat com base nos embeddings gerados. Aqui, você pode interagir com o modelo em tempo real, utilizando um fluxo de perguntas e respostas.

- **main.py**: Script que executa a aplicação de chat como uma interface web usando Streamlit. Ele permite que usuários finais façam consultas diretamente em uma interface amigável.

## Execução do Projeto

Para a execução do projeto, instale o [Docker](https://docs.docker.com/engine/install/) juntamente com o Docker Compose.

Acesse a raiz do projeto no terminal e execute o comando

```
docker compose up
```

Caso não possua placa de vídeo Nvidia, comente ou apague as partes de deploy no arquivo `docker-compose.yaml`

### Streamlit

Assim que a aplicação iniciar, acesse em seu navegador o endereço:
`http://localhost:3011/`

Clique em `Gerar Novos Embeddings`, escolha o arquivo para ser analisado, e inicie a conversa com a aplicação, conforme o [exemplo](/assets/streamlit.webm).

### Jupyter notebook

Para testar o notebook, veja o token gerado para o servidor do jupyter com o comando

```
docker compose logs jupyter
```

Acesse o endereço `http://localhost:8888/`
Cole o token de acesso e faça login, escolha o arquivo `jupyter.ipynb`, e execute o passo a passo.

## Prompting no RAG

A técnica de prompting utilizada é o **Chain-of-Thought Prompting**, que é particularmente eficaz para elicitar raciocínio em LLMs. Esta abordagem consiste em fornecer ao modelo uma cadeia de raciocínio, o que melhora o desempenho em tarefas complexas como raciocínio aritmético, senso comum e manipulação simbólica.

Baseado em experimentos documentados, como o artigo ["Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903), essa técnica foi usada para obter ganhos de desempenho notáveis em tarefas de raciocínio.
