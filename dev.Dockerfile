FROM python:3

WORKDIR /app

COPY requirements.txt ./

RUN curl https://ollama.ai/install.sh | sh
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c" , "ollama serve & sleep 5 && ollama pull llama3.2 && streamlit run main.py"]
