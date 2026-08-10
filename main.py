from fastapi import FastAPI
from src.app.api import app
if __name__ == "__main__":
    import uvicorn
    app = FastAPI()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)