# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

# copy requirements trước để tận dụng cache
COPY requirements.txt .

# Install build deps, install python deps (dùng cache pip), then purge build deps in the same RUN
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      gcc \
      curl \
      git \
 && pip install --upgrade pip setuptools wheel \
 # ưu tiên binary nếu có, dùng cache mount cho pip
 && pip install --no-cache-dir --prefer-binary -r requirements.txt \
 # remove build deps to keep image small
 && apt-get purge -y --auto-remove build-essential gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy source
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "front_end.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false"]
