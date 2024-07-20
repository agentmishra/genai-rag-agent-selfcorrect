
from pprint import pprint
from typing import List
import json
from langchain_core.documents import Document
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv

from langgraph.graph import END, StateGraph, START
import requests
from flask import Flask, request,jsonify
from waitress import serve
load_dotenv()

SEARCH_SERVICE = os.getenv('SEARCH_SERVICE')

### State

class HttpRequestHandler:
    def __init__(self, url, method, keys):
        self.url = url
        self.method = method
        self.keys = keys

    def fetch(self, data=None, headers=None):
        try:
            if self.method.upper() == 'GET':
                response = requests.get(self.url,json=data, headers=headers)
            elif self.method.upper() == 'POST':
                response = requests.post(self.url, json=data, headers=headers)
            elif self.method.upper() == 'PUT':
                response = requests.put(self.url, json=data, headers=headers)
            elif self.method.upper() == 'DELETE':
                response = requests.delete(self.url, headers=headers)
            else:
                raise ValueError("Unsupported HTTP method")
            
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            json_response = response.json()
            
            return {key: json_response.get(key, None) for key in self.keys}
        
        except requests.exceptions.RequestException as e:
            if e.response is not None:
               try:
                    error_response = e.response.json()
                    print("Error response JSON:", error_response)
               except ValueError as ve:
                   print("Error response text:", e.response.text)
            else:
                print(f"HTTP request failed: {e}")
            return None
        

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval

    url = "http://docReader:8080/collection/search"
    method = "GET"
    keys = ["result"]

    handler = HttpRequestHandler(url, method, keys)
    response_data = handler.fetch({"query": question, "maxResults": 2})
    documents = response_data['result'] #retriever.invoke(question)
    print(response_data)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    url = "http://llm:8080/generate"
    method = "POST"
    keys = ["result"]

    handler = HttpRequestHandler(url, method, keys)
    response_data = handler.fetch({"context": documents, "prompt": question})
    print(response_data)
    # RAG generation
    generation =response_data['result'] # rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    url = "http://docGrader:8080/grade/promptToDocument"
    method = "POST"
    keys = ["grade"]
    handler = HttpRequestHandler(url, method, keys)

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        response_data = handler.fetch({"prompt": question , "doc": d})        
        score = response_data['grade']
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search 
    print("Web search  -- NOT IMPLEMENTED YET --")
    url = "http://webSearch:8080/search"
    method = "POST"
    keys = ["result"]

    handler = HttpRequestHandler(url, method, keys)
    response_data = handler.fetch({"query": question})     
    if not response_data:
        return {"documents": documents, "question": question}
    docs = response_data["result"] # web_search_tool.invoke({"query": question})
    print(f"search results {response_data["result"]}")
    web_results=''
    if SEARCH_SERVICE == 'duckduck':
        web_results = "\n".join([d["body"] for d in docs])
    elif SEARCH_SERVICE == 'tavily':
        web_results = "\n".join([d["content"] for d in docs])

    print(f"search results {web_results}")
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

### Conditional edge

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    url = "http://router:8080/generateRoutePath"
    method = "POST"
    keys = ["result"]

    handler = HttpRequestHandler(url, method, keys)
    response_data = handler.fetch({"prompt": question})    
    source = json.loads(response_data['result'])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

### Conditional edge

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    print(f"generation is {generation}")
    if generation == None :
        generation = ''
    url = "http://docGrader:8080/grade/generationToDocuments"
    method = "POST"
    keys = ["grade"]

    handler = HttpRequestHandler(url, method, keys)
    response_data = handler.fetch({"docs": documents, "generation": generation})    
    score = response_data["grade"] 
    grade = score["score"]
    print(f"NOT halucination grade {grade}")
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        url = "http://docGrader:8080/grade/promptToGeneration"
        handler = HttpRequestHandler(url, method, keys)
        response_data = handler.fetch({"prompt": question, "generation": generation})    
        QaScore = response_data["grade"]   # answer_grader.invoke({"question": question, "generation": generation})
        QaGrade = QaScore["score"]
        if QaGrade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve from docReader
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve", #  retrieve from docReader
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile
app = workflow.compile()
fApp = Flask(__name__)

@fApp.route('/generate', methods=['POST'])
def gen():
    data = request.get_json()
    prompt = data.get('prompt')
    if prompt == '' or prompt == None: 
        return jsonify({'error': 'prompt is empty'})    
    inputs = {"question": prompt}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    print(value["generation"])
    return jsonify({ "generation": value["generation"]})


if __name__ == '__main__':  
    print('ready') 
    serve(fApp, host="0.0.0.0", port=8080)