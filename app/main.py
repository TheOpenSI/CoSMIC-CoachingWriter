from fastapi import FastAPI
from .api.routes import router

app = FastAPI(title="CoSMIC Coaching Writer", version="0.1.0")

app.include_router(router)


@app.get("/")
def root():
    return {"service": "CoSMIC-CoachingWriter", "message": "Academic writing coaching service.", "docs": "/docs"}
