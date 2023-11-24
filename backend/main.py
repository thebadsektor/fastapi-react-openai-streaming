from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import logging
import sys
import os
from llama_index.llms import OpenAI
import openai
import io
import json
from pydantic import BaseModel
import asyncio
from typing import Generator

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
)

from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Set up CORS middleware
origins = [
    "http://localhost:3000",
    "localhost:3000"
]

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# Set up OpenAI API key
client = openai_api_key = os.environ.get("OPENAI_API_KEY")

# I'm using a cheap GPT-3 model for this demo
llm = OpenAI(model="gpt-3.5-turbo")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

service_context = ServiceContext.from_defaults(
            chunk_size=500,
            chunk_overlap=20,
            llm=llm,
            embed_model=embed_model,
        )

class SearchRequest(BaseModel):
    query: str

# Global storage for search results
search_results = []

@app.post("/search")
async def search(request_data: SearchRequest):  # request_data will be a SearchRequest instance
    
    global search_results

    query = request_data.query
    # query = "write a short story."
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)  # Assuming docs are already loaded
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    search_results = [line for line in response.response]

    return response.response

search("write a short story")

search_results = ['a','b','c']

@app.get("/stream")
async def stream_data() -> StreamingResponse:
    async def event_stream() -> Generator:
        for line in search_results:
            yield f"data: {line}".encode('utf-8')
            # await asyncio.sleep(0.1)  # Adjust delay as needed

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Mount the React build folder as a static files directory
app.mount("/",
          StaticFiles(directory="frontend/build", html=True),
          name="static")