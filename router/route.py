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
from langchain_core.output_parsers import StrOutputParser

# Access the variables
database_url = os.getenv('DATABASE_URL')
local_llm = os.getenv('LLM_ID')
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
os.environ["USER_AGENT"] = "llm service"
# Using ollama as docker service
# Note ollama docker image has istructions how to use nvidia toolkit with docker!
# llm = ChatOllama(base_url='ollama:11434',model=local_llm, format="json", temperature=0)
llm = ChatOllama(base_url=LLM_BASE_URL,model=local_llm,  temperature=0)
# Embeddings
 
app = Flask(__name__)

# Chroma DB client
client = chromadb.HttpClient(host='vectorDB', port=8000,settings=Settings(allow_reset=True))
 
# Create a new collection or get an existing one
collection_name = "rag-chroma"
collection = client.get_or_create_collection(collection_name) #, embedding_function=default_ef


@app.route('/generateRoutePath', methods=['POST'])
def generateRoutePath():
    data = request.get_json()
    prompt = data.get('prompt')

    print(prompt)
    print(local_llm)


    if prompt == '' or prompt == None: 
        return jsonify({'error': 'prompt is empty'})    
    promptTmpl = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
    rag_chain = promptTmpl | llm | StrOutputParser()
    generation = rag_chain.invoke({"question": prompt})
    print(generation)
    return jsonify({"llm": "online", "result": generation})
 
if __name__ == '__main__':  
    print('ready') 
    serve(app, host="0.0.0.0", port=8080)



