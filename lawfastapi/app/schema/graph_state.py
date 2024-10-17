from pydantic import BaseModel
from typing import TypedDict

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    relevance: str
    session_id: str

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str
    # context: str
