# Bibo — Backend Trợ Lý Ảo AI Tiếng Việt Cho Trẻ Em (6–10 Tuổi)

Backend viết bằng **Python + FastAPI + Google Gemini API (`google-genai`)** xây dựng trợ lý ảo **Bibo** an toàn, vui vẻ, giàu tính giáo dục cho trẻ em Việt Nam từ 6 đến 10 tuổi.

## Tính năng chính

1. **Endpoint `POST /chat`**: Nhận văn bản tiếng Việt (kèm `session_id` tùy chọn để giữ ngữ cảnh hội thoại nhiều lượt) và trả về câu trả lời thân thiện từ Bibo.
2. **Bảo vệ trẻ em tối đa (`BLOCK_LOW_AND_ABOVE`)**: Cấu hình `safety_settings` ở ngưỡng chặn cao nhất trên cả 4 danh mục an toàn (`HARASSMENT`, `HATE_SPEECH`, `SEXUALLY_EXPLICIT`, `DANGEROUS_CONTENT`), kèm phản hồi dự phòng nhẹ nhàng khi nội dung bị chặn.
3. **Tính cách giáo dục (`system_instruction`)**: Định hình trợ lý ảo **Bibo** trả lời ngắn gọn, dễ hiểu, vui vẻ, tuyệt đối tránh chủ đề người lớn, bạo lực, chính trị và khéo léo hướng bé sang học tập.
4. **Function Calling (Tools) tự động** với 3 hàm:
   - `calculate_math`: Tính toán biểu thức toán học an toàn bằng cây cú pháp AST (không dùng `eval()`).
   - `get_current_time`: Lấy thứ, ngày/tháng/năm và giờ hiện tại theo giờ Việt Nam (`Asia/Ho_Chi_Minh`) hoặc múi giờ khác.
   - `search_vietnamese_wikipedia`: Tra cứu và tóm tắt kiến thức từ Wikipedia tiếng Việt (`vi.wikipedia.org`).

## Cấu trúc thư mục

```text
.
├── main.py              # Khởi tạo FastAPI và các endpoint (/, /health, /chat)
├── config.py            # Biến môi trường, system_instruction, safety_settings
├── tools.py             # 3 hàm Function Calling (toán học, thời gian, Wikipedia)
├── gemini_service.py    # Kết nối Google Gemini API, quản lý session & bộ lọc an toàn
├── requirements.txt     # Danh sách thư viện Python cần cài đặt
├── .env.example         # File mẫu biến môi trường
├── pytest.ini           # Cấu hình kiểm thử pytest
└── tests/
    └── test_app.py      # Bộ test tự động cho tools, safety settings và /chat
```

## Hướng dẫn cài đặt & chạy

### 1. Tạo môi trường ảo Python (khuyến nghị Python 3.10+)

```bash
python3 -m venv .venv

# Trên Linux / macOS:
source .venv/bin/activate

# Trên Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Key

Lấy Gemini API Key tại [Google AI Studio](https://aistudio.google.com/apikey), sau đó tạo file `.env` từ `.env.example`:

```bash
cp .env.example .env
```

Mở file `.env` và điền khóa của bạn:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
```

> **Lưu ý quan trọng về model `gemini-1.5-flash`:**
> Code giữ mặc định `GEMINI_MODEL=gemini-1.5-flash` đúng theo yêu cầu ban đầu. Tuy nhiên, Google đã chính thức ngừng hỗ trợ dòng `gemini-1.5-flash` trên Gemini API từ ngày **29/09/2025**. Khi chạy thực tế, nếu API trả lỗi `404` không tìm thấy model, bạn chỉ cần đổi `GEMINI_MODEL` trong file `.env` sang model Flash mới đang hoạt động (ví dụ `GEMINI_MODEL=gemini-flash-latest` hoặc `GEMINI_MODEL=gemini-3.8-flash`) mà **không cần sửa code**.

### 4. Khởi chạy Backend FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi chạy, mở trình duyệt tại `http://localhost:8000/docs` để thử trực tiếp trên Swagger UI.

## Ví dụ gọi API `/chat`

```bash
# Hỏi phép toán (gọi tool calculate_math)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bibo ơi, 15 nhân 4 cộng 8 bằng bao nhiêu?", "session_id": "be-an"}'

# Hỏi giờ hiện tại (gọi tool get_current_time)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bây giờ là mấy giờ rồi Bibo?", "session_id": "be-an"}'

# Tra cứu Wikipedia tiếng Việt (gọi tool search_vietnamese_wikipedia)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Mặt Trời là gì vậy Bibo?", "session_id": "be-an"}'
```

## Chạy kiểm thử tự động

```bash
pytest -v
```
