from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.API import chat_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["https://localhost:8000", "http://localhost:8000", "http://localhost:3000"]
)

app.include_router(chat_router, tags=["Chat"], prefix="/api")