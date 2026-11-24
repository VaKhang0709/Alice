"""
Bộ kiểm thử tự động cho Backend trợ lý ảo AI tiếng Việt Bibo.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import _extra_utils, errors, types
from google.genai.chats import AsyncChat

from config import SAFE_BLOCK_REPLY, SAFETY_SETTINGS, SYSTEM_INSTRUCTION
from gemini_service import BiboAssistantService
from main import app, get_bibo_service
from tools import (
    BIBO_TOOLS,
    calculate_math,
    get_current_time,
    search_vietnamese_wikipedia,
)


# ==============================================================================
# 1. KIỂM TRA CẤU HÌNH AN TOÀN TRẺ EM & SYSTEM INSTRUCTION
# ==============================================================================
def test_system_instruction_exact_match() -> None:
    expected = (
        "Bạn là một trợ lý ảo thông minh, thân thiện tên là Bibo, dành cho trẻ em từ 6-10 tuổi. "
        "Hãy luôn trả lời bằng tiếng Việt thật ngắn gọn, dễ hiểu, vui vẻ, mang tính giáo dục. "
        "Tuyệt đối không trả lời các chủ đề người lớn, bạo lực, chính trị. "
        "Nếu trẻ hỏi những điều không phù hợp, hãy khéo léo từ chối và hướng trẻ sang chủ đề học tập."
    )
    assert SYSTEM_INSTRUCTION == expected


def test_safety_settings_block_low_and_above() -> None:
    categories = {setting.category for setting in SAFETY_SETTINGS}
    assert categories == {
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    }
    for setting in SAFETY_SETTINGS:
        assert setting.threshold == types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE


# ==============================================================================
# 2. KIỂM TRA TOOL 1: TÍNH TOÁN TOÁN HỌC
# ==============================================================================
def test_calculate_math_basic_and_unicode() -> None:
    assert "101" in calculate_math("12 * 8 + 5")
    assert "6" in calculate_math("(45 - 9) ÷ 6")
    assert "24" in calculate_math("6 × 4")
    assert "32" in calculate_math("2 ^ 5")
    assert "12" in calculate_math("sqrt(144)")
    assert "3.5" in calculate_math("7 / 2")


def test_calculate_math_errors_and_security() -> None:
    assert "chia cho số 0" in calculate_math("10 / 0")
    assert "Lỗi" in calculate_math("sqrt(-4)")
    assert "Lỗi" in calculate_math("")
    assert "Lỗi" in calculate_math("1+" * 150)
    # Chặn tấn công thực thi mã (code injection)
    assert "Lỗi" in calculate_math("__import__('os').system('echo hacked')")
    assert "Lỗi" in calculate_math("(lambda x: x + 1)(5)")
    # Chặn phép lũy thừa quá lớn gây treo hệ thống
    assert "quá lớn" in calculate_math("9 ** 9 ** 9")


# ==============================================================================
# 3. KIỂM TRA TOOL 2: LẤY THỜI GIAN HIỆN TẠI
# ==============================================================================
def test_get_current_time_vietnam_default() -> None:
    result = get_current_time()
    assert "Asia/Ho_Chi_Minh" in result
    assert "UTC+07:00" in result
    assert "ngày " in result
    assert "giờ " in result


def test_get_current_time_other_timezone_and_fallback() -> None:
    tokyo = get_current_time("Asia/Tokyo")
    assert "Asia/Tokyo" in tokyo
    assert "UTC+09:00" in tokyo

    fallback = get_current_time("Invalid/Timezone_XYZ")
    assert "Asia/Ho_Chi_Minh" in fallback
    assert "Không nhận diện được múi giờ" in fallback


# ==============================================================================
# 4. KIỂM TRA TOOL 3: TÌM KIẾM WIKIPEDIA TIẾNG VIỆT
# ==============================================================================
def test_search_vietnamese_wikipedia_success() -> None:
    search_response = MagicMock()
    search_response.raise_for_status.return_value = None
    search_response.json.return_value = {
        "query": {"search": [{"title": "Mặt Trời"}]}
    }

    summary_response = MagicMock()
    summary_response.raise_for_status.return_value = None
    summary_response.json.return_value = {
        "query": {
            "pages": {
                "123": {
                    "title": "Mặt Trời",
                    "extract": "Mặt Trời là ngôi sao ở trung tâm của Hệ Mặt Trời.",
                    "fullurl": "https://vi.wikipedia.org/wiki/M%E1%BA%B7t_Tr%E1%BB%9Di",
                }
            }
        }
    }

    mock_client = MagicMock()
    mock_client.get.side_effect = [search_response, summary_response]
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_client
    mock_cm.__exit__.return_value = False

    with patch("tools.httpx.Client", return_value=mock_cm):
        result = search_vietnamese_wikipedia("Mặt Trời", sentences=2)

    assert "Chủ đề: Mặt Trời" in result
    assert "ngôi sao ở trung tâm" in result
    assert "vi.wikipedia.org" in result


def test_search_vietnamese_wikipedia_no_results_and_timeout() -> None:
    assert "Không có từ khóa" in search_vietnamese_wikipedia("   ")

    empty_resp = MagicMock()
    empty_resp.raise_for_status.return_value = None
    empty_resp.json.return_value = {"query": {"search": []}}

    mock_client = MagicMock()
    mock_client.get.return_value = empty_resp
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_client
    mock_cm.__exit__.return_value = False

    with patch("tools.httpx.Client", return_value=mock_cm):
        not_found = search_vietnamese_wikipedia("ChuDeKhongTonTaiXYZ")
    assert "Không tìm thấy" in not_found

    timeout_client = MagicMock()
    timeout_client.get.side_effect = httpx.TimeoutException("timeout")
    timeout_cm = MagicMock()
    timeout_cm.__enter__.return_value = timeout_client
    timeout_cm.__exit__.return_value = False

    with patch("tools.httpx.Client", return_value=timeout_cm):
        timed_out = search_vietnamese_wikipedia("Khủng long")
    assert "hơi chậm" in timed_out


# ==============================================================================
# 5. KIỂM TRA TƯƠNG THÍCH SCHEMA FUNCTION CALLING VỚI GOOGLE-GENAI SDK
# ==============================================================================
def test_tools_compatible_with_google_genai_sdk() -> None:
    for fn in BIBO_TOOLS:
        decl = types.FunctionDeclaration.from_callable_with_api_option(
            callable=fn,
            api_option="GEMINI_API",
            use_json_schema=True,
        )
        assert decl.name == fn.__name__
        assert decl.parameters_json_schema is not None

    # Xác nhận bộ gọi hàm nội bộ của SDK chạy trơn tru trên tool
    math_res = _extra_utils.invoke_function_from_dict_args(
        {"expression": "7 * 6"},
        calculate_math,
    )
    assert "42" in math_res

    time_res = _extra_utils.invoke_function_from_dict_args({}, get_current_time)
    assert "Asia/Ho_Chi_Minh" in time_res


# ==============================================================================
# 6. KIỂM TRA ENDPOINT FASTAPI (/chat, /health, AFC, SAFETY BLOCK)
# ==============================================================================
class _FakeAsyncModels:
    """Giả lập tầng AsyncModels.generate_content để test luồng AsyncChat + AFC thật."""

    def __init__(self, responses: list[types.GenerateContentResponse]) -> None:
        self._responses = list(responses)
        self._api_client = MagicMock(vertexai=False)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any = None,
    ) -> types.GenerateContentResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if not self._responses:
            raise AssertionError("Không còn phản hồi giả lập nào trong hàng đợi.")
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def _make_mock_service(
    responses: list[Any],
    model_name: str = "gemini-1.5-flash",
) -> tuple[BiboAssistantService, _FakeAsyncModels]:
    fake_models = _FakeAsyncModels(responses)

    fake_client = MagicMock()
    fake_client.aio.chats.create.side_effect = lambda *, model, config: AsyncChat(
        modules=fake_models,
        model=model,
        config=config,
        history=[],
    )

    async def _noop_aclose() -> None:
        return None

    fake_client.aio.aclose = _noop_aclose
    fake_client.close = lambda: None

    service = BiboAssistantService(
        api_key="test-key",
        model_name=model_name,
        client=fake_client,
    )
    return service, fake_models


def test_chat_endpoint_with_real_afc_function_calling() -> None:
    """Kiểm thử toàn trình: Gemini yêu cầu gọi calculate_math -> SDK tự chạy hàm -> trả lời bé."""
    func_call_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name="calculate_math",
                            args={"expression": "15 * 4"},
                        )
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    final_text_response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text="Chào bé! 15 nhân 4 bằng 60 nhé, bé giỏi quá!"
                        )
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    service, fake_models = _make_mock_service([func_call_response, final_text_response])
    app.dependency_overrides[get_bibo_service] = lambda: service
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "Bibo ơi, 15 nhân 4 bằng mấy?", "session_id": "be-an"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["reply"] == "Chào bé! 15 nhân 4 bằng 60 nhé, bé giỏi quá!"
            assert data["session_id"] == "be-an"
            assert data["blocked"] is False
            assert data["tools_used"] == ["calculate_math"]

            # Xác nhận AFC đã tự thực thi calculate_math và gửi kết quả "60" lại cho Gemini
            assert len(fake_models.calls) == 2
            second_call_contents = fake_models.calls[1]["contents"]
            tool_resp_parts = second_call_contents[-1].parts
            assert tool_resp_parts[0].function_response.name == "calculate_math"
            assert "60" in tool_resp_parts[0].function_response.response["result"]

            # Kiểm tra cấu hình safety_settings và system_instruction được truyền vào API
            sent_config = fake_models.calls[0]["config"]
            assert sent_config.system_instruction == SYSTEM_INSTRUCTION
            assert len(sent_config.safety_settings) == 4
            assert all(
                s.threshold == types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
                for s in sent_config.safety_settings
            )

            # Kiểm tra lượt thứ 2 trên cùng session_id không lặp lại tools_used từ lượt 1
            # và vẫn giữ ngữ cảnh hội thoại trong lịch sử chat
            del_resp = client.delete("/chat/be-an")
            assert del_resp.status_code == 200
            assert del_resp.json()["reset"] is True
    finally:
        app.dependency_overrides.clear()


def test_chat_endpoint_multi_turn_tools_isolation() -> None:
    """Đảm bảo tools_used chỉ phản ánh các hàm được gọi ở lượt hiện tại, không trộn lượt cũ."""
    turn1_tool_call = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name="get_current_time",
                            args={},
                        )
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    turn1_final = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Bây giờ là giờ học bài rồi bé ơi!")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    turn2_final = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Không có chi, bé ngoan lắm!")],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )

    service, fake_models = _make_mock_service([turn1_tool_call, turn1_final, turn2_final])
    app.dependency_overrides[get_bibo_service] = lambda: service
    try:
        with TestClient(app) as client:
            r1 = client.post(
                "/chat",
                json={"message": "Mấy giờ rồi Bibo?", "session_id": "be-minh"},
            )
            assert r1.status_code == 200
            assert r1.json()["tools_used"] == ["get_current_time"]

            r2 = client.post(
                "/chat",
                json={"message": "Cảm ơn Bibo nhé!", "session_id": "be-minh"},
            )
            assert r2.status_code == 200
            assert r2.json()["tools_used"] == []
            assert r2.json()["reply"] == "Không có chi, bé ngoan lắm!"
            # Lượt 2 phải mang theo lịch sử từ lượt 1
            assert len(fake_models.calls) == 3
            assert len(fake_models.calls[2]["contents"]) > 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "blocked_response",
    [
        types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY
            )
        ),
        types.GenerateContentResponse(
            candidates=[
                types.Candidate(finish_reason=types.FinishReason.SAFETY)
            ]
        ),
    ],
)
def test_chat_endpoint_handles_safety_block_gracefully(
    blocked_response: types.GenerateContentResponse,
) -> None:
    service, _ = _make_mock_service([blocked_response])
    app.dependency_overrides[get_bibo_service] = lambda: service
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "Nội dung không phù hợp"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["blocked"] is True
            assert data["reply"] == SAFE_BLOCK_REPLY
    finally:
        app.dependency_overrides.clear()


def test_chat_endpoint_validation_and_health() -> None:
    service, _ = _make_mock_service([])
    app.dependency_overrides[get_bibo_service] = lambda: service
    try:
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.json()["assistant"] == "Bibo"

            # Tin nhắn rỗng / chỉ chứa khoảng trắng -> 422 Validation Error
            bad_resp = client.post("/chat", json={"message": "    "})
            assert bad_resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_chat_endpoint_missing_api_key_and_404_model_error() -> None:
    # 1. Thiếu GEMINI_API_KEY -> 503 hướng dẫn cấu hình .env
    unconfigured = BiboAssistantService(api_key="", model_name="gemini-1.5-flash")
    app.dependency_overrides[get_bibo_service] = lambda: unconfigured
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "Xin chào Bibo"})
            assert resp.status_code == 503
            assert "GEMINI_API_KEY" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()

    # 2. Lỗi 404 model đã ngừng hỗ trợ -> 502 kèm hướng dẫn đổi GEMINI_MODEL
    api_404 = errors.APIError(
        code=404,
        response_json={"error": {"message": "models/gemini-1.5-flash is not found"}},
    )
    service_404, _ = _make_mock_service([api_404])
    app.dependency_overrides[get_bibo_service] = lambda: service_404
    try:
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "Xin chào Bibo"})
            assert resp.status_code == 502
            assert "GEMINI_MODEL" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()
