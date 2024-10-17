from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.main_router import router
from langchain_teddynote import logging as langsmith_logging
import uvicorn
import sys
import os


def init():
    # LangSmith 로깅 설정
    langsmith_logging.langsmith("Law-RAG-API")
    
    # FastAPI 설정
    fastapi_app = FastAPI()

    # CORS 설정
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Router 설정
    fastapi_app.include_router(router)
    return fastapi_app


# FastAPI 실행
if __name__ == "__main__":
    fastapi_app = init()
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
else:
    init()
