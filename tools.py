"""
Các hàm công cụ (Function Calling / Tools) tích hợp cho trợ lý ảo Bibo.

Google Gemini sẽ đọc tên hàm, kiểu dữ liệu tham số và docstring của từng hàm bên dưới
để tự động quyết định khi nào cần gọi công cụ nào nhằm trả lời chính xác cho trẻ em.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from config import DEFAULT_TIMEZONE, WIKIPEDIA_TIMEOUT_SECONDS

# ==============================================================================
# CÁC HẰNG SỐ PHỤC VỤ TÍNH TOÁN AN TOÀN (KHÔNG DÙNG eval() TRỰC TIẾP)
# ==============================================================================
_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_NODES = 100
_MAX_EXPONENT = 100
_MAX_ABS_RESULT = 1e100

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_MATH_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "min": min,
    "max": max,
}

_ALLOWED_MATH_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_VIETNAMESE_WEEKDAYS: dict[int, str] = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật",
}

_WIKIPEDIA_API_URL = "https://vi.wikipedia.org/w/api.php"
_WIKIPEDIA_HEADERS = {
    "User-Agent": "BiboKidsAssistant/1.0 (Vietnamese Educational AI Assistant)",
    "Accept": "application/json",
}


def _safe_eval_ast(node: ast.AST) -> float:
    """Đánh giá đệ quy cây cú pháp toán học AST một cách an toàn tuyệt đối."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError("Chỉ chấp nhận giá trị số.")

    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in _ALLOWED_MATH_CONSTANTS:
            return _ALLOWED_MATH_CONSTANTS[key]
        raise ValueError(f"Không hỗ trợ hằng số hoặc biến '{node.id}'.")

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError("Phép toán một ngôi không được hỗ trợ.")
        return op_func(_safe_eval_ast(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError("Phép toán không được hỗ trợ.")
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Không thể chia cho 0.")

        if isinstance(node.op, ast.Pow):
            # Chặn phép lũy thừa quá lớn gây treo CPU hoặc tràn bộ nhớ
            if abs(right) > _MAX_EXPONENT:
                raise OverflowError("Số mũ quá lớn.")
            if left == 0 and right < 0:
                raise ZeroDivisionError("Không thể chia cho 0.")
            if left < 0 and not float(right).is_integer():
                raise ValueError("Không hỗ trợ lũy thừa phân số của số âm.")

        result = op_func(left, right)
        if not math.isfinite(result) or abs(result) > _MAX_ABS_RESULT:
            raise OverflowError("Kết quả quá lớn.")
        return float(result)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("Lời gọi hàm không hợp lệ.")
        func_name = node.func.id.lower()
        func = _ALLOWED_MATH_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"Không hỗ trợ hàm '{node.func.id}'.")
        args = [_safe_eval_ast(arg) for arg in node.args]
        if func_name == "pow" and len(args) == 2 and abs(args[1]) > _MAX_EXPONENT:
            raise OverflowError("Số mũ quá lớn.")
        if func_name == "round" and len(args) == 2:
            result = round(args[0], int(args[1]))
        else:
            result = func(*args)
        if not math.isfinite(result) or abs(result) > _MAX_ABS_RESULT:
            raise OverflowError("Kết quả quá lớn.")
        return float(result)

    raise ValueError("Biểu thức chứa thành phần không được phép.")


def _format_number(value: float) -> str:
    """Định dạng số đẹp mắt cho trẻ em (bỏ phần .0 thừa nếu là số nguyên)."""
    if math.isclose(value, round(value), rel_tol=1e-12, abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ==============================================================================
# TOOL 1: TÍNH TOÁN TOÁN HỌC
# ==============================================================================
def calculate_math(expression: str) -> str:
    """Tính toán giá trị của một biểu thức toán học chính xác và an toàn.

    Sử dụng công cụ này khi bé hỏi các phép tính cộng, trừ, nhân, chia,
    lũy thừa, căn bậc hai hoặc bài toán số học.

    Args:
        expression: Biểu thức toán học cần tính, ví dụ '12 * 8 + 5',
            '(45 - 9) / 6', '2 ** 5', hoặc 'sqrt(144)'.

    Returns:
        Chuỗi tiếng Việt chứa kết quả phép tính hoặc thông báo lỗi thân thiện.
    """
    raw = (expression or "").strip()
    if not raw:
        return "Lỗi: Biểu thức toán học đang bị trống."
    if len(raw) > _MAX_EXPRESSION_LENGTH:
        return "Lỗi: Biểu thức toán học quá dài, hãy chia nhỏ phép tính nhé."

    # Chuẩn hóa các ký hiệu toán học phổ biến sang cú pháp Python
    normalized = (
        raw.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("^", "**")
    )

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return f"Lỗi: Không thể hiểu biểu thức '{raw}'. Hãy kiểm tra lại phép tính nhé."

    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        return "Lỗi: Biểu thức quá phức tạp, hãy chia thành từng bước nhỏ nhé."

    try:
        value = _safe_eval_ast(tree)
    except ZeroDivisionError:
        return "Lỗi: Trong toán học không thể chia cho số 0 được đâu nhé!"
    except OverflowError:
        return "Lỗi: Kết quả của phép tính này quá lớn để hiển thị."
    except ValueError as exc:
        return f"Lỗi: Không thể tính '{raw}' ({exc})."

    formatted = _format_number(value)
    return f"Kết quả của {raw} là {formatted}."


# ==============================================================================
# TOOL 2: LẤY THỜI GIAN HIỆN TẠI
# ==============================================================================
def _vietnamese_period_of_day(hour: int) -> str:
    """Xác định buổi trong ngày theo thói quen tiếng Việt."""
    if 5 <= hour < 11:
        return "buổi sáng"
    if 11 <= hour < 13:
        return "buổi trưa"
    if 13 <= hour < 18:
        return "buổi chiều"
    if 18 <= hour < 22:
        return "buổi tối"
    return "đêm khuya"


def get_current_time(timezone_name: str = DEFAULT_TIMEZONE) -> str:
    """Lấy ngày, thứ và giờ hiện tại theo múi giờ chỉ định (mặc định giờ Việt Nam).

    Sử dụng công cụ này khi bé hỏi bây giờ là mấy giờ, hôm nay là thứ mấy,
    ngày tháng năm hiện tại hoặc giờ ở một quốc gia/thành phố khác.

    Args:
        timezone_name: Tên múi giờ chuẩn IANA, mặc định là 'Asia/Ho_Chi_Minh'
            (giờ Việt Nam, UTC+7). Ví dụ khác: 'Asia/Tokyo', 'Europe/London'.

    Returns:
        Chuỗi tiếng Việt mô tả rõ giờ, phút, buổi, thứ và ngày/tháng/năm hiện tại.
    """
    tz_key = (timezone_name or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    fallback_note = ""
    try:
        tz = ZoneInfo(tz_key)
    except ZoneInfoNotFoundError:
        tz_key = DEFAULT_TIMEZONE
        tz = ZoneInfo(DEFAULT_TIMEZONE)
        fallback_note = (
            f" (Không nhận diện được múi giờ '{timezone_name}', "
            f"đã chuyển sang giờ Việt Nam {DEFAULT_TIMEZONE}.)"
        )

    now = datetime.now(tz)
    weekday = _VIETNAMESE_WEEKDAYS[now.weekday()]
    period = _vietnamese_period_of_day(now.hour)
    utc_offset = now.strftime("%z")
    utc_label = f"UTC{utc_offset[:3]}:{utc_offset[3:]}" if len(utc_offset) == 5 else "UTC"

    return (
        f"Hiện tại theo múi giờ {tz_key} ({utc_label}) là "
        f"{now.hour} giờ {now.minute:02d} phút {now.second:02d} giây ({period}), "
        f"{weekday}, ngày {now.day:02d}/{now.month:02d}/{now.year}."
        f"{fallback_note}"
    )


# ==============================================================================
# TOOL 3: TÌM KIẾM THÔNG TIN TRÊN WIKIPEDIA TIẾNG VIỆT
# ==============================================================================
def search_vietnamese_wikipedia(query: str, sentences: int = 3) -> str:
    """Tìm kiếm và tóm tắt kiến thức phổ thông từ Wikipedia tiếng Việt.

    Sử dụng công cụ này khi bé hỏi kiến thức về khoa học, động vật, thực vật,
    thiên văn, địa lý, lịch sử, danh nhân hoặc các hiện tượng tự nhiên.

    Args:
        query: Từ khóa hoặc chủ đề cần tìm bằng tiếng Việt, ví dụ 'Mặt Trời',
            'Khủng long', 'Vịnh Hạ Long', 'Vòng tuần hoàn của nước'.
        sentences: Số câu tóm tắt muốn lấy từ bài viết (từ 1 đến 5, mặc định 3).

    Returns:
        Chuỗi tiếng Việt gồm tiêu đề bài viết, đoạn tóm tắt ngắn và nguồn Wikipedia,
        hoặc thông báo khi không tìm thấy kết quả.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return "Không có từ khóa để tìm kiếm trên Wikipedia tiếng Việt."

    clamped_sentences = max(1, min(int(sentences), 5))

    try:
        with httpx.Client(
            headers=_WIKIPEDIA_HEADERS,
            timeout=WIKIPEDIA_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            # Bước 1: Tìm các bài viết phù hợp nhất trên Wikipedia tiếng Việt
            search_resp = client.get(
                _WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": clean_query,
                    "srlimit": 3,
                    "utf8": 1,
                    "format": "json",
                },
            )
            search_resp.raise_for_status()
            search_results = search_resp.json().get("query", {}).get("search", [])

            if not search_results:
                return (
                    f"Không tìm thấy bài viết nào về '{clean_query}' "
                    "trên Wikipedia tiếng Việt."
                )

            # Bước 2: Lấy đoạn mở đầu dạng văn bản thuần từ bài viết phù hợp đầu tiên có nội dung
            for item in search_results:
                title = item.get("title", "").strip()
                if not title:
                    continue

                summary_resp = client.get(
                    _WIKIPEDIA_API_URL,
                    params={
                        "action": "query",
                        "prop": "extracts|info",
                        "inprop": "url",
                        "exintro": 1,
                        "explaintext": 1,
                        "exsentences": clamped_sentences,
                        "redirects": 1,
                        "titles": title,
                        "format": "json",
                    },
                )
                summary_resp.raise_for_status()
                pages = summary_resp.json().get("query", {}).get("pages", {})

                for page in pages.values():
                    extract = (page.get("extract") or "").strip()
                    if not extract:
                        continue
                    page_title = page.get("title", title)
                    page_url = page.get("fullurl", "")
                    # Giới hạn độ dài để giữ câu trả lời ngắn gọn, dễ hiểu với trẻ em
                    if len(extract) > 1200:
                        extract = extract[:1197].rstrip() + "..."
                    source_suffix = f"\nNguồn: {page_url}" if page_url else ""
                    return f"Chủ đề: {page_title}\nNội dung tóm tắt: {extract}{source_suffix}"

            return (
                f"Đã tìm thấy chủ đề liên quan đến '{clean_query}' nhưng chưa có "
                "đoạn tóm tắt phù hợp trên Wikipedia tiếng Việt."
            )

    except httpx.TimeoutException:
        return "Kết nối tới Wikipedia tiếng Việt đang hơi chậm, bé thử hỏi lại sau nhé."
    except httpx.HTTPError:
        return "Hiện chưa thể kết nối tới Wikipedia tiếng Việt để tra cứu."
    except Exception:
        return "Đã xảy ra lỗi nhỏ khi tra cứu Wikipedia tiếng Việt."


# Danh sách tất cả các công cụ (Function Calling) cấp cho Gemini
BIBO_TOOLS = [
    calculate_math,
    get_current_time,
    search_vietnamese_wikipedia,
]
