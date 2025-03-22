from fastapi import FastAPI
from .routers.prediction import router

app = FastAPI()

app.include_router(router, tags=["prediction"])