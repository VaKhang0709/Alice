"""
Ứng dụng FastAPI Backend cho trợ lý ảo AI tiếng Việt Bibo (dành cho trẻ em 6-10 tuổi).

Tính năng chính:
- Endpoint POST /chat nhận văn bản tiếng Việt và trả lời ngắn gọn, vui vẻ, giáo dục.
- Bộ lọc an toàn trẻ em mức cao nhất (BLOCK_LOW_AND_ABOVE) từ Google Gemini API.
- Tích hợp Function Calling (Tools): tính toán toán học, xem giờ hiện tại,
  và tra cứu Wikipedia tiếng Việt.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config import MAX_MESSAGE_LENGTH
from gemini_service import BiboAssistantService, BiboServiceError


# ==============================================================================
# 1. VÒNG ĐỜI ỨNG DỤNG (LIFESPAN) & DEPENDENCY INJECTION
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Khởi tạo và dọn dẹp tài nguyên dịch vụ Gemini cùng vòng đời FastAPI."""
    service = BiboAssistantService()
    app.state.bibo_service = service
    try:
        yield
    finally:
        await service.aclose()


def get_bibo_service(request: Request) -> BiboAssistantService:
    """Dependency cung cấp instance BiboAssistantService cho các route."""
    return request.app.state.bibo_service


# ==============================================================================
# 2. KHỞI TẠO ỨNG DỤNG FASTAPI
# ==============================================================================
app = FastAPI(
    title="Bibo - Trợ lý ảo AI tiếng Việt cho trẻ em (6-10 tuổi)",
    description=(
        "Backend FastAPI tích hợp Google Gemini API với bộ lọc an toàn "
        "BLOCK_LOW_AND_ABOVE và Function Calling (toán học, thời gian, Wikipedia tiếng Việt)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Bật CORS để Frontend (Web / Mobile app) dễ dàng kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 3. CÁC SCHEMA PYDANTIC (ĐẦU VÀO / ĐẦU RA)
# ==============================================================================
class ChatRequest(BaseModel):
    """Dữ liệu đầu vào cho endpoint /chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Câu hỏi hoặc tin nhắn bằng tiếng Việt từ bé.",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Mã phiên trò chuyện (tùy chọn). Gửi lại session_id nhận được từ lượt trước "
            "để Bibo nhớ ngữ cảnh cuộc hội thoại."
        ),
    )

    @field_validator("message")
    @classmethod
    def validate_non_blank_message(cls, value: str) -> str:
        """Loại bỏ khoảng trắng thừa và từ chối tin nhắn chỉ gồm khoảng trắng."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Tin nhắn không được để trống.")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Bibo ơi, 25 nhân 4 cộng 10 bằng mấy và bây giờ là mấy giờ rồi?",
                    "session_id": "be-an-001",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Dữ liệu đầu ra trả về từ endpoint /chat."""

    reply: str = Field(..., description="Câu trả lời bằng tiếng Việt từ trợ lý ảo Bibo.")
    session_id: str = Field(..., description="Mã phiên hội thoại hiện tại.")
    blocked: bool = Field(
        default=False,
        description="True nếu nội dung bị bộ lọc an toàn trẻ em chặn và chuyển sang câu từ chối nhẹ nhàng.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Danh sách các hàm công cụ (Function Calling) Bibo đã sử dụng để trả lời.",
    )


class HealthResponse(BaseModel):
    """Thông tin trạng thái hoạt động của Backend."""

    status: str
    assistant: str
    model: str
    api_key_configured: bool


# ==============================================================================
# 4. CÁC ENDPOINT API
# ==============================================================================
@app.get("/", tags=["Hệ thống"])
async def root() -> dict[str, str]:
    """Trang chào mừng và đường dẫn nhanh tới tài liệu API."""
    return {
        "message": "Chào mừng đến với Backend trợ lý ảo Bibo dành cho trẻ em 6-10 tuổi!",
        "chat_endpoint": "POST /chat",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Hệ thống"])
async def health_check(
    service: BiboAssistantService = Depends(get_bibo_service),
) -> HealthResponse:
    """Kiểm tra tình trạng hoạt động và cấu hình model hiện tại."""
    return HealthResponse(
        status="ok",
        assistant="Bibo",
        model=service.model_name,
        api_key_configured=service.is_configured,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Trò chuyện"])
async def chat_with_bibo(
    payload: ChatRequest,
    service: BiboAssistantService = Depends(get_bibo_service),
) -> ChatResponse:
    """Nhận câu hỏi tiếng Việt của trẻ em và trả về câu trả lời an toàn từ Bibo.

    Bibo có thể tự động gọi các công cụ:
    - `calculate_math`: Tính toán biểu thức toán học.
    - `get_current_time`: Xem ngày và giờ hiện tại.
    - `search_vietnamese_wikipedia`: Tra cứu kiến thức trên Wikipedia tiếng Việt.
    """
    try:
        result = await service.chat(
            message=payload.message,
            session_id=payload.session_id,
        )
    except BiboServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return ChatResponse(
        reply=result.reply,
        session_id=result.session_id,
        blocked=result.blocked,
        tools_used=result.tools_used,
    )


@app.delete("/chat/{session_id}", tags=["Trò chuyện"])
async def reset_chat_session(
    session_id: str,
    service: BiboAssistantService = Depends(get_bibo_service),
) -> dict[str, str | bool]:
    """Xóa lịch sử hội thoại của một phiên trò chuyện để bắt đầu lại từ đầu."""
    removed = service.reset_session(session_id)
    return {
        "session_id": session_id,
        "reset": removed,
        "message": (
            "Đã làm mới cuộc trò chuyện với Bibo."
            if removed
            else "Không tìm thấy phiên trò chuyện cần xóa."
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
