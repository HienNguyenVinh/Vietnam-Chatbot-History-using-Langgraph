from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from API import cart_router, chat_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["https://localhost:3000", "http://localhost:3000"]
)

app.include_router(chat_router, tags=["Chat"], prefix="/api")