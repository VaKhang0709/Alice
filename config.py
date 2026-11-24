"""
Cấu hình trung tâm cho Backend trợ lý ảo AI tiếng Việt Bibo (dành cho trẻ em 6-10 tuổi).
"""

import os
from dotenv import load_dotenv
from google.genai import types

# Tự động nạp các biến môi trường từ file .env (nếu có)
load_dotenv()

# ==============================================================================
# 1. CẤU HÌNH GOOGLE GEMINI API
# ==============================================================================
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# Model mặc định theo yêu cầu: gemini-1.5-flash.
# LƯU Ý: Google đã ngừng cung cấp gemini-1.5-flash trên Gemini API từ 29/09/2025.
# Bạn có thể đổi nhanh qua biến môi trường GEMINI_MODEL trong file .env (không cần sửa code),
# ví dụ: GEMINI_MODEL=gemini-flash-latest hoặc GEMINI_MODEL=gemini-3.8-flash.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

# ==============================================================================
# 2. SYSTEM INSTRUCTION (ĐỊNH HÌNH TÍNH CÁCH & QUY TẮC AN TOÀN CHO BIBO)
# ==============================================================================
SYSTEM_INSTRUCTION: str = (
    "Bạn là một trợ lý ảo thông minh, thân thiện tên là Bibo, dành cho trẻ em từ 6-10 tuổi. "
    "Hãy luôn trả lời bằng tiếng Việt thật ngắn gọn, dễ hiểu, vui vẻ, mang tính giáo dục. "
    "Tuyệt đối không trả lời các chủ đề người lớn, bạo lực, chính trị. "
    "Nếu trẻ hỏi những điều không phù hợp, hãy khéo léo từ chối và hướng trẻ sang chủ đề học tập."
)

# ==============================================================================
# 3. SAFETY SETTINGS Ở MỨC CAO NHẤT (BLOCK_LOW_AND_ABOVE) ĐỂ BẢO VỆ TRẺ EM
# ==============================================================================
# Áp dụng ngưỡng chặn nghiêm ngặt nhất (BLOCK_LOW_AND_ABOVE) trên cả 4 danh mục
# an toàn được Gemini Developer API hỗ trợ:
# - HARASSMENT: quấy rối, bắt nạt, xúc phạm
# - HATE_SPEECH: ngôn từ thù ghét, phân biệt đối xử
# - SEXUALLY_EXPLICIT: nội dung tình dục, nội dung người lớn
# - DANGEROUS_CONTENT: bạo lực, vũ khí, tự gây hại, hành vi nguy hiểm
SAFETY_SETTINGS: list[types.SafetySetting] = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]

# Câu trả lời dự phòng thân thiện khi bộ lọc an toàn của Gemini chặn nội dung
SAFE_BLOCK_REPLY: str = (
    "Ôi, chủ đề này chưa phù hợp với lứa tuổi của bé đâu nhé! "
    "Bibo luôn muốn giữ an toàn cho bé. Hay là chúng mình cùng giải một bài toán vui, "
    "xem bây giờ là mấy giờ, hoặc khám phá điều thú vị về động vật và vũ trụ nhé?"
)

# ==============================================================================
# 4. CẤU HÌNH ỨNG DỤNG & GIỚI HẠN BẢO VỆ HỆ THỐNG
# ==============================================================================
MAX_MESSAGE_LENGTH: int = 1000       # Giới hạn độ dài tin nhắn đầu vào
MAX_CHAT_SESSIONS: int = 500         # Số phiên hội thoại lưu tối đa trong bộ nhớ
MAX_TOOL_CALL_ROUNDS: int = 5        # Giới hạn số vòng gọi công cụ tự động (AFC) mỗi lượt
WIKIPEDIA_TIMEOUT_SECONDS: float = 8.0
DEFAULT_TIMEZONE: str = "Asia/Ho_Chi_Minh"
