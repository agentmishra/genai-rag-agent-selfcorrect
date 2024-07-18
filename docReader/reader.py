from dotenv import load_dotenv
import os
import sys
import uuid
from flask import Flask, request,jsonify
from waitress import serve
import chromadb

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Access the variables
database_url = os.getenv('DATABASE_URL')
local_llm = os.getenv('LLM_ID')
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
os.environ["USER_AGENT"] = "reader service"
# Using ollama as docker service
# Note ollama docker image has istructions how to use nvidia toolkit with docker!
# llm = ChatOllama(base_url='ollama:11434',model=local_llm, format="json", temperature=0)
llm = ChatOllama(base_url=LLM_BASE_URL,model=local_llm, format="json", temperature=0)
# Embeddings
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

app = Flask(__name__)

def loadDocs():
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


# Chroma DB client
client = chromadb.HttpClient(host='vectorDB', port=8000,settings=Settings(allow_reset=True))
# resets the database
# client.reset()  

embedding_model = NomicEmbeddings(
    model="nomic-embed-text-v1.5", 
    inference_mode="local",
    dimensionality=384
    )
# Create a new collection or get an existing one
collection_name = "rag-chroma"
collection = client.get_or_create_collection(collection_name) #, embedding_function=default_ef


@app.route('/collection/reset', methods=['POST'])
def reset():
    client.reset()  
    collection = client.get_or_create_collection(collection_name) 
    return jsonify({"docReader": "online", "result": 'Collection truncated!'})

@app.route('/collection/populate', methods=['POST'])
def populate():
    collection = client.get_or_create_collection(collection_name)
    doc_splits = loadDocs()
    for doc in doc_splits:
        rand = str(uuid.uuid1())
        collection.add(ids=[rand], metadatas=doc.metadata, documents=doc.page_content)
    total_items =  len(collection.get()["ids"]) 
    return jsonify({"docReader": "online", "result": f"Collection populated, total documents: {total_items}"})

if __name__ == '__main__':  
    print('ready') 
    serve(app, host="0.0.0.0", port=8080)



