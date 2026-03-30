# 1. Gunakan Python versi ringan
FROM python:3.10-slim

# 2. Set folder kerja di dalam server
WORKDIR /app

# 3. Copy requirements dan install library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy semua file project kamu ke server
COPY . .

# 5. Jalankan FastAPI di port 7860 (WAJIB untuk Hugging Face)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]