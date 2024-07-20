## Self correct RAG implementation


### Services
* vectorDB
* router
* webSearch
* docReader
* docGrader 
* llm


### docReader

| Endpoint    | Method    | Description    | Params |
|---------------- | --------------- | --------------- |--------------- |
| /collection/reset   | POST   | Truncated document collection    | - |
| /collection/populate   | POST   | Populates document collection with fresh fetched data from target sources. | - |
| /collection/search | GET | Do semantic search through the vector db | JSON [ query , maxResults ] |

### docGrader

| Endpoint    | Method    | Description    | Params |
|---------------- | --------------- | --------------- |--------------- |
| /grade/promptToDocument | POST  | Searches in vector db against user submited prompt, then llm will evalute how relative is the user prompt with best match from db (if any). | JSON [ prompt ] |
| /grade/promptToGeneration   | POST   | Evaluates the llm response against user prompt    | JSON [ prompt, generation] |
| /grade/generationToDocuments | POST | Evalutes if model answer is grounded to provided docs | JSON [ docs , generation ] |


 

