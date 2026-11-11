FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOCK_MODE=0 \
    MODEL_DIR=/mnt/models \
    PORT=9000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG INSTALL_MODELS=0
COPY requirements.txt requirements-models.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_MODELS" = "1" ]; then pip install --no-cache-dir -r requirements-models.txt; fi \
    && python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY app ./app
COPY README.md ./

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
