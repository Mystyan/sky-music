FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOCK_MODE=0 \
    MODEL_DIR=/mnt/models \
    PORT=9000 \
    OMP_NUM_THREADS=8 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# --- CUDA user-space libs for torch, installed in small layers ---
# (ACR personal edition rejects very large single blobs, so keep layers < ~1GB)
RUN pip install --no-cache-dir --no-deps \
    nvidia-cublas-cu12==12.4.5.8 \
    nvidia-cuda-runtime-cu12==12.4.127 \
    nvidia-cuda-nvrtc-cu12==12.4.127 \
    nvidia-cuda-cupti-cu12==12.4.127 \
    nvidia-nvtx-cu12==12.4.127 \
    sympy==1.13.1 networkx jinja2 fsspec filelock typing-extensions

RUN pip install --no-cache-dir --no-deps nvidia-cudnn-cu12==9.1.0.70

RUN pip install --no-cache-dir --no-deps \
    nvidia-cufft-cu12==11.2.1.3 \
    nvidia-curand-cu12==10.3.5.147 \
    nvidia-nvjitlink-cu12==12.4.127

RUN pip install --no-cache-dir --no-deps \
    nvidia-cusolver-cu12==11.6.1.9 \
    nvidia-cusparse-cu12==12.3.1.170

RUN pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.21.5

RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1

# --- application dependencies ---
RUN pip install --no-cache-dir fastapi 'uvicorn[standard]' python-multipart \
    'pydantic>=2.7,<3' numpy scipy soundfile mido httpx imageio-ffmpeg

RUN pip install --no-cache-dir librosa pretty_midi

RUN pip install --no-cache-dir demucs

RUN pip install --no-cache-dir basic-pitch

# 预下载 Demucs 权重，避免冷启动联网
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY app ./app
COPY vendor ./vendor
COPY README.md ./

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
