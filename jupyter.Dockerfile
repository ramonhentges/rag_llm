FROM python:3

WORKDIR /app

COPY requirements.txt ./

RUN curl https://ollama.ai/install.sh | sh
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter

ENTRYPOINT ["/bin/sh", "-c" , "ollama serve & sleep 5 && ollama pull llama3.2 && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
