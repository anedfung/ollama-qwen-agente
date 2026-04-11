from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.agent import preguntar_agente

app = FastAPI()

# Define allowed origins
origins = [
"http://localhost",
"http://localhost:3000",
"https://yourdomain.com"
]

app.add_middleware(
CORSMiddleware,
allow_origins=origins, # List of allowed origins
allow_credentials=True,
allow_methods=["*"], # Allow all HTTP methods
allow_headers=["*"], # Allow all headers
)

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    respuesta = preguntar_agente(query.question)
    return {"response": respuesta}