"""
Lớp dịch vụ kết nối Google Gemini API cho trợ lý ảo Bibo.

Đảm nhiệm:
- Khởi tạo Google GenAI client với cấu hình an toàn trẻ em (BLOCK_LOW_AND_ABOVE),
  system_instruction và 3 hàm Function Calling (Tools).
- Quản lý phiên hội thoại nhiều lượt (session) có khóa đồng bộ hóa và giới hạn bộ nhớ.
- Phát hiện nội dung bị bộ lọc an toàn của Gemini chặn để trả về câu từ chối nhẹ nhàng.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from google import genai
from google.genai import errors, types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_CHAT_SESSIONS,
    MAX_TOOL_CALL_ROUNDS,
    SAFE_BLOCK_REPLY,
    SAFETY_SETTINGS,
    SYSTEM_INSTRUCTION,
)
from tools import BIBO_TOOLS

logger = logging.getLogger(__name__)

# Các lý do kết thúc cho thấy nội dung đã bị chặn vì vi phạm an toàn
_BLOCKED_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.MODEL_ARMOR,
    types.FinishReason.IMAGE_SAFETY,
    types.FinishReason.IMAGE_PROHIBITED_CONTENT,
}


class BiboServiceError(Exception):
    """Lỗi nghiệp vụ / kết nối Gemini với mã HTTP và thông báo thân thiện."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ChatResult:
    """Kết quả trả về sau mỗi lượt trò chuyện với Bibo."""

    reply: str
    session_id: str
    blocked: bool = False
    tools_used: list[str] = field(default_factory=list)


class BiboAssistantService:
    """Dịch vụ quản lý trợ lý ảo Bibo sử dụng Google Gemini API (SDK google-genai)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else GEMINI_API_KEY).strip()
        self.model_name = (model_name if model_name is not None else GEMINI_MODEL).strip()
        self._client: Optional[genai.Client] = client

        # Cấu hình chuẩn dùng chung cho mọi phiên trò chuyện của Bibo
        self._generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            safety_settings=SAFETY_SETTINGS,
            tools=BIBO_TOOLS,
            temperature=0.7,
            max_output_tokens=512,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALL_ROUNDS,
            ),
        )

        # Lưu cache các phiên chat dạng LRU (session_id -> (chat_session, lock))
        self._sessions: OrderedDict[str, tuple[Any, asyncio.Lock]] = OrderedDict()

    @property
    def is_configured(self) -> bool:
        """Kiểm tra đã có client hoặc GEMINI_API_KEY hợp lệ hay chưa."""
        return self._client is not None or bool(self.api_key)

    @property
    def generation_config(self) -> types.GenerateContentConfig:
        """Trả về cấu hình Gemini hiện tại (phục vụ kiểm tra / kiểm thử)."""
        return self._generation_config

    def _get_client(self) -> genai.Client:
        """Khởi tạo Google GenAI Client theo cơ chế lazy-loading."""
        if self._client is None:
            if not self.api_key:
                raise BiboServiceError(
                    "Chưa cấu hình GEMINI_API_KEY. Hãy tạo file .env từ .env.example "
                    "và điền khóa API lấy tại https://aistudio.google.com/apikey.",
                    status_code=503,
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _get_or_create_session(self, session_id: str) -> tuple[Any, asyncio.Lock]:
        """Lấy phiên chat hiện có hoặc tạo phiên AsyncChat mới kèm khóa đồng bộ."""
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        client = self._get_client()
        chat = client.aio.chats.create(
            model=self.model_name,
            config=self._generation_config,
        )
        lock = asyncio.Lock()
        self._sessions[session_id] = (chat, lock)

        # Giới hạn số phiên lưu trong bộ nhớ để tránh rò rỉ RAM
        while len(self._sessions) > MAX_CHAT_SESSIONS:
            self._sessions.popitem(last=False)

        return chat, lock

    def reset_session(self, session_id: str) -> bool:
        """Xóa lịch sử của một phiên trò chuyện."""
        return self._sessions.pop(session_id, None) is not None

    async def aclose(self) -> None:
        """Đóng kết nối HTTP nền khi tắt ứng dụng FastAPI."""
        self._sessions.clear()
        if self._client is not None:
            try:
                await self._client.aio.aclose()
            except Exception:
                logger.debug("Bỏ qua lỗi khi đóng AsyncClient.", exc_info=True)
            try:
                self._client.close()
            except Exception:
                logger.debug("Bỏ qua lỗi khi đóng Client.", exc_info=True)

    @staticmethod
    def _is_safety_blocked(response: types.GenerateContentResponse) -> bool:
        """Kiểm tra câu hỏi đầu vào hoặc câu trả lời đầu ra có bị chặn an toàn không."""
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if (
                block_reason is not None
                and block_reason != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
            ):
                return True

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason in _BLOCKED_FINISH_REASONS:
                return True

        return False

    @staticmethod
    def _extract_tools_used(
        chat: Any,
        prev_history_len: int,
        response: types.GenerateContentResponse,
    ) -> list[str]:
        """Thu thập danh sách tên các hàm công cụ đã được Gemini gọi trong lượt này."""
        called_names: list[str] = []

        def _collect_from_contents(contents: Optional[list[types.Content]]) -> None:
            if not contents:
                return
            for content in contents:
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    func_call = getattr(part, "function_call", None)
                    if func_call and getattr(func_call, "name", None):
                        if func_call.name not in called_names:
                            called_names.append(func_call.name)

        # Kiểm tra lịch sử chi tiết mới thêm vào chat session trong lượt này
        get_history = getattr(chat, "get_history", None)
        if callable(get_history):
            full_history = get_history(curated=False) or []
            _collect_from_contents(full_history[prev_history_len:])

        # Kiểm tra thêm automatic_function_calling_history (nếu SDK trả về)
        afc_history = getattr(response, "automatic_function_calling_history", None)
        _collect_from_contents(afc_history)

        return called_names

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> ChatResult:
        """Gửi tin nhắn tiếng Việt tới Gemini và nhận phản hồi an toàn từ Bibo."""
        sid = (session_id or "").strip() or str(uuid.uuid4())
        chat_session, lock = self._get_or_create_session(sid)

        async with lock:
            get_history = getattr(chat_session, "get_history", None)
            prev_history_len = (
                len(get_history(curated=False) or []) if callable(get_history) else 0
            )

            try:
                # Google GenAI SDK sẽ tự động xử lý vòng lặp Function Calling (AFC):
                # 1. Gửi câu hỏi lên Gemini cùng danh sách 3 hàm tools.
                # 2. Nếu Gemini yêu cầu gọi hàm (toán / giờ / Wikipedia), SDK tự thực thi hàm.
                # 3. Gửi kết quả hàm ngược lại cho Gemini để Bibo diễn giải dễ hiểu cho bé.
                response = await chat_session.send_message(message)
            except errors.APIError as exc:
                logger.exception("Lỗi từ Google Gemini API: %s", exc)
                status = getattr(exc, "code", None)
                if status == 404:
                    raise BiboServiceError(
                        f"Model '{self.model_name}' không khả dụng trên Gemini API (404). "
                        "Lưu ý: Google đã ngừng hỗ trợ 'gemini-1.5-flash' từ 29/09/2025. "
                        "Hãy cập nhật GEMINI_MODEL trong file .env sang model Flash mới "
                        "(ví dụ: gemini-flash-latest hoặc gemini-3.8-flash).",
                        status_code=502,
                    ) from exc
                if status in (401, 403):
                    raise BiboServiceError(
                        "GEMINI_API_KEY không hợp lệ hoặc chưa được cấp quyền truy cập.",
                        status_code=401,
                    ) from exc
                if status == 429:
                    raise BiboServiceError(
                        "Bibo đang hơi bận vì có nhiều câu hỏi cùng lúc (vượt giới hạn API). "
                        "Bé đợi một chút rồi hỏi lại nhé!",
                        status_code=429,
                    ) from exc
                raise BiboServiceError(
                    "Không thể kết nối tới Google Gemini API lúc này. Vui lòng thử lại sau.",
                    status_code=502,
                ) from exc

            tools_used = self._extract_tools_used(chat_session, prev_history_len, response)

            if self._is_safety_blocked(response):
                return ChatResult(
                    reply=SAFE_BLOCK_REPLY,
                    session_id=sid,
                    blocked=True,
                    tools_used=tools_used,
                )

            reply_text = (getattr(response, "text", None) or "").strip()
            if not reply_text:
                # Nếu mô hình không trả về text (ví dụ bị lọc âm thầm), trả lời dự phòng an toàn
                return ChatResult(
                    reply=SAFE_BLOCK_REPLY,
                    session_id=sid,
                    blocked=True,
                    tools_used=tools_used,
                )

            return ChatResult(
                reply=reply_text,
                session_id=sid,
                blocked=False,
                tools_used=tools_used,
            )
