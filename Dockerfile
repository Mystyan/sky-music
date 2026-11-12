FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOCK_MODE=0 \
    MODEL_DIR=/tmp/models \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/root/.cache/huggingface \
    PORT=9000 \
    OMP_NUM_THREADS=8 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# SubRoformer 的分频前端需要在运行期编译 C++ 扩展，镜像里必须带编译器
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# --- 基础数值/服务依赖 ---
RUN pip install --no-cache-dir \
    "numpy>=2,<3" scipy soundfile \
    fastapi "uvicorn[standard]" python-multipart "pydantic>=2.7,<3" httpx mido imageio-ffmpeg

# --- CUDA user-space libs for torch，拆成小层（ACR 单次上传会话有时限）---
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

# torch 本体拆成 3 个层安装
RUN pip download --no-cache-dir --no-deps -d /tmp/w --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 \
 && python -c "import glob,zipfile; z=zipfile.ZipFile(glob.glob('/tmp/w/torch-*.whl')[0]); z.extract('torch/lib/libtorch_cuda.so','/usr/local/lib/python3.11/site-packages')" \
 && rm -rf /tmp/w

RUN pip download --no-cache-dir --no-deps -d /tmp/w --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 \
 && python -c "import glob,zipfile; z=zipfile.ZipFile(glob.glob('/tmp/w/torch-*.whl')[0]); z.extract('torch/lib/libtorch_cpu.so','/usr/local/lib/python3.11/site-packages')" \
 && rm -rf /tmp/w

RUN pip download --no-cache-dir --no-deps -d /tmp/w --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 \
 && python -c "import glob,zipfile; w=zipfile.ZipFile(glob.glob('/tmp/w/torch-*.whl')[0]); skip={'torch/lib/libtorch_cuda.so','torch/lib/libtorch_cpu.so'}; [w.extract(n,'/usr/local/lib/python3.11/site-packages') for n in w.namelist() if n not in skip]" \
 && rm -rf /tmp/w

# --- 音频分析 ---
RUN pip install --no-cache-dir librosa pretty_midi

# --- 低档 Basic Pitch（不装 TensorFlow/TFLite，走 ONNX）---
RUN pip install --no-cache-dir --no-deps basic-pitch==0.4.0 \
 && pip install --no-cache-dir mir-eval resampy scikit-learn onnxruntime

# --- 低档 Demucs（锁 torchaudio，避免 pip 连带升级 torch）---
RUN pip install --no-cache-dir "torchaudio==2.5.1" demucs

# --- 中/高档 MuScriptor（权重首次调用时下载到 MODEL_DIR）---
RUN pip install --no-cache-dir muscriptor

# --- 中/高档 SubRoformer 分轨（Python 3.11）---
RUN pip install --no-cache-dir \
    "https://github.com/ArchiMickey/SubRoformer/archive/refs/heads/main.tar.gz"

# 预下载 SubRoformer 权重并预编译 C++ 扩展，避免运行期现场编译
RUN python -c "from subroformer.inference.engine import load_separator; load_separator(device='cpu'); print('subroformer ready')"

# --- 高档 VocalParse 人声转谱 ---
RUN pip install --no-cache-dir --no-deps \
    "https://github.com/pymaster17/VocalParse/archive/refs/heads/main.tar.gz" \
 && pip install --no-cache-dir "qwen-asr>=0.0.6" "transformers>=4.50" "accelerate>=1.12.0" safetensors pyarrow

# 预下载 Demucs 权重，避免冷启动联网
# 注：dl.fbaipublicfiles.com 证书已过期，这里跳过证书校验下载（模型文件仍由 torch 的 hash 校验）
RUN python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; from demucs.pretrained import get_model; get_model('htdemucs')"

COPY app ./app
COPY vendor ./vendor
COPY README.md ./

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
