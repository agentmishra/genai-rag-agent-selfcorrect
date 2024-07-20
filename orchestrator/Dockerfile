FROM python:3.12.4-slim
ARG PATH_TO_COPY=.

RUN pip install -U  torch python-dotenv langchain-nomic langchain_community tiktoken langchainhub chromadb langchain \
    langgraph tavily-python nomic[local] langchain-text-splitters    

RUN pip install -U bs4 uuid waitress flask langchain_core typing_extensions
# Install Node.js and npm
RUN curl -sS https://deb.nodesource.com/setup_22.x | bash -
RUN apt-get update && apt-get install -y nodejs npm && npm i -g nodemon
# Set environment variable for Node.js
ENV NODE_PATH /usr/local/lib/node_modules/
WORKDIR /app
COPY ${PATH_TO_COPY} .

# CMD ["python","-u", "/app/reader.py"]
 
