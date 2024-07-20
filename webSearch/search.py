from dotenv import load_dotenv
import os
import sys
import uuid
from flask import Flask, request,jsonify
from waitress import serve
import chromadb
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults

from duckduckgo_search import DDGS

SEARCH_SERVICE = os.getenv('SEARCH_SERVICE')

app = Flask(__name__)
web_search_tool = TavilySearchResults(k=3)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')   
    results=[]
    if SEARCH_SERVICE == "duckduck":
        results = DDGS().text(
            keywords=query,
            region='wt-wt',
            safesearch='off',
            timelimit='30d',
            max_results=2
        )
    elif SEARCH_SERVICE == 'tavily':
        results =  web_search_tool.invoke({"query": query})
    return jsonify({ "result": results})




if __name__ == '__main__':  
    print('ready') 
    
    serve(app, host="0.0.0.0", port=8080)
