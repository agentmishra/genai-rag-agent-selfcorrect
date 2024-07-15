FROM python:3.12.4-slim

RUN pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain \
    langgraph tavily-python nomic[local] langchain-text-splitters

WORKDIR /app

COPY . ./app