# 基礎映像 (選擇適合你的 Python 版本)
FROM python:3.9-slim-buster

# 工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt requirements.txt

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式程式碼
COPY . .

# 暴露端口 (根據你的 Flask 設定)
EXPOSE 5000

# 啟動命令
CMD ["python", "app.py"]