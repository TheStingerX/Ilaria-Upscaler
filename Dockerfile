# CPU-oriented reference Dockerfile. For Ultra, build a CUDA image separately.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 appuser && mkdir -p /app/models/quality && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
