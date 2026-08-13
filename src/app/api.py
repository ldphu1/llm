import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.routes import rag_chain

app = FastAPI(title="LangChain RAG SSE API")

# Cấu hình CORS để cho phép Web Frontend (React, Vue, HTML) gọi tới API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

async def event_generator(question: str):
    """Generator bất đồng bộ đóng gói dữ liệu theo chuẩn SSE"""
    try:
        # astream() giúp đọc token async từ LCEL chain
        async for chunk in rag_chain.astream(question):
            # Định dạng chuẩn SSE: data: \n\n
            payload = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
    except Exception as e:
        error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {error_payload}\n\n"
    finally:
        # Báo hiệu cho client biết luồng đã kết thúc
        yield "data: [DONE]\n\n"

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Endpoint nhận câu hỏi và trả về Stream SSE"""
    return StreamingResponse(
        event_generator(request.question),
        media_type="text/event-stream"
    )