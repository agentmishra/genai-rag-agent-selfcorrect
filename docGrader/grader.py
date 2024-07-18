from dotenv import load_dotenv
import os
from flask import Flask, request,jsonify
from waitress import serve
import chromadb

# Load environment variables from .env file
load_dotenv()

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from chromadb.config import Settings


# Access the variables
database_url = os.getenv('DATABASE_URL')
local_llm = os.getenv('LLM_ID')
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
os.environ["USER_AGENT"] = "grader service"
# Using ollama as docker service
# Note ollama docker image has istructions how to use nvidia toolkit with docker!
# llm = ChatOllama(base_url='ollama:11434',model=local_llm, format="json", temperature=0)
llm = ChatOllama(base_url=LLM_BASE_URL,model=local_llm, format="json", temperature=0)
app = Flask(__name__)

# Chroma DB client
client = chromadb.HttpClient(host='vectorDB', port=8000,settings=Settings(allow_reset=True))

# Create a new collection or get an existing one
collection_name = "rag-chroma"
collection = client.get_or_create_collection(collection_name) #, embedding_function=default_ef

@app.route('/grade/promptToDocument', methods=['POST']) # Retrieval Grader
def grade():
    collection = client.get_or_create_collection(collection_name) #, embedding_function=default_ef    
    data = request.get_json()
    prompt = data.get('prompt')
    if prompt == '' or prompt == None: 
        return jsonify({'error': 'prompt is empty'})
    print(f"prompt-> {prompt}")
    results = collection.query(
        query_texts=[prompt],
        n_results=1, 
        include=['documents']
    )
    doc_txt = results['documents'][0]
    promptTempl = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    retrieval_grader = promptTempl | llm | JsonOutputParser()
    grade = retrieval_grader.invoke({"question": prompt, "document": doc_txt});
    print(grade)      
    return jsonify({"docGrader": "online", "grade": grade})

@app.route('/grade/promptToGeneration', methods=['POST']) # Answer Grader
def gradePromptToGeneration():
    data = request.get_json()
    prompt = data.get('prompt')
    generation = data.get('generation')
    if generation == '' or generation == None: 
        return jsonify({'error': 'generation is empty'})
    if prompt == '' or prompt == None: 
        return jsonify({'error': 'prompt is empty'})
    print(f"prompt-> {prompt}")    
    promptTempl = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    answer_grader = promptTempl | llm | JsonOutputParser()
    grade = answer_grader.invoke({"question": prompt, "generation": generation});
    print(grade)      
    return jsonify({"docGrader": "online", "grade": grade})

if __name__ == '__main__':  
    print('ready') 
    serve(app, host="0.0.0.0", port=8080)



