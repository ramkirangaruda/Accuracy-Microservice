from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="AI Testcase Validator"
)

app.include_router(router)