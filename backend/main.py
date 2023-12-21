from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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
from typing import Generator, List
from dotenv import load_dotenv

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
load_dotenv()
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
    chunk_size: int = 5  # Default to 1 for per-token streaming

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, chunk_size: int = 1):
    await manager.connect(websocket)
    try:
        while True:
            query = await websocket.receive_text()
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
            streaming_response = query_engine.query(query)

            buffer = ""
            buffer_length = 0

            for text in streaming_response.response_gen:
                buffer += text + " "
                buffer_length += 1

                if chunk_size > 0 and buffer_length >= chunk_size:
                    await manager.send_personal_message(buffer.strip(), websocket)
                    buffer = ""
                    buffer_length = 0
                elif chunk_size == 0:
                    await manager.send_personal_message(text, websocket)

            if buffer:
                await manager.send_personal_message(buffer.strip(), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Mount the React build folder as a static files directory
app.mount("/",
          StaticFiles(directory="frontend/build", html=True),
          name="static")