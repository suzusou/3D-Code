# Pythonのベースイメージを指定
FROM python:3.11.7-slim

# 作業ディレクトリを作成
WORKDIR /app

# 必要なツールと依存ライブラリをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# requirements.txt をコピー
COPY requirements.txt .

# Pythonライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY app /app/app

# 環境変数にTesseractのパスを指定
ENV TESSERACT_CMD=/usr/bin/tesseract

# 必要なポートを公開
EXPOSE xxxx

# アプリケーションを起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "xxxx"]

#build:
    #context: .
    #dockerfile:Dockerfile
