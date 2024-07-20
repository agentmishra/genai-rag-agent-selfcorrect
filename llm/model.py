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


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')
    context = data.get('context')
    print(prompt)
    print(local_llm)

    if context == '' or context == None: 
        return jsonify({'error': 'context is empty'})
    if prompt == '' or prompt == None: 
        return jsonify({'error': 'prompt is empty'})    
    promptTmpl = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )
    try:
        rag_chain = promptTmpl | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": context, "question": prompt})
        print(generation)
        return jsonify({"llm": "online", "result": generation})
    except Exception as e: 
        print(f"exception -->{e}")
        response = jsonify({
            "error": "Bad request",
            "details": "The request could not be understood by the server due to malformed syntax",
            "status": "error"})
        if "Ollama call failed with status code 404" in f"{e}":
            response = jsonify({
            "error": "Ollama server seems is missing the specified model ({local_llm})",
            "details": "Try to pull it in ollama server by : ollama pull {local_llm}",
            "status": "Model not found"})
        
        response.status_code = 400
        return response

 
if __name__ == '__main__':  
    print('ready') 
    serve(app, host="0.0.0.0", port=8080)



