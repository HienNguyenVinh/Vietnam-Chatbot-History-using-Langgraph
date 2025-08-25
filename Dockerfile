FROM python:3.11-slim

WORKDIR /app

# Cài gói hệ thống cần thiết
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy và cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "front_end.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false"]
