# sky-music

把一首歌的音频，串成一条“分轨 → 人声/伴奏转谱 → 节奏/调性/和弦 → 光遇编排 → SkyStudio/ABC15”的工作流。默认 `MOCK_MODE=1`，没有 GPU、没有模型权重也能启动并跑通接口；设置 `MOCK_MODE=0` 并挂载模型后再启用真实模型。

## 工作流

```text
audio
  → ffmpeg 归一化
  → SubRoformer / Demucs 分轨
  → VocalParse / GAME 人声转谱
  → MuScriptor 多乐器转谱
  → librosa 节拍 + 调性
  → 和弦推断
  → 三档编排（低 / 中 / 高）
  → SkyStudio JSON + ABC15 TXT
```

## 三档模型质量

| 档位 | 分轨 | 人声 | 多乐器 | 排布 |
|---|---|---|---|---|
| 低 `low` | Demucs | GAME-small | MuScriptor-small | 单音，强拍根音 |
| 中 `medium` | SubRoformer | GAME-medium | MuScriptor-medium | 双音，每两拍 |
| 高 `high` | SubRoformer | VocalParse-1.7B + GAME-large | MuScriptor-large | 三音，每拍 |

三档都保证：音域 C4–C6、同键间隔 ≥70ms、输出只包含 15 个光遇键位。模型缺失时会回退并在 `warnings` 中说明。

## 本地运行

```bash
pip install -r requirements.txt
MOCK_MODE=1 uvicorn app.main:app --host 0.0.0.0 --port 9000
```

打开 `http://127.0.0.1:9000/docs`，用 `/api/transcribe` 上传音频，或用（`quality` 取 `low` / `medium` / `high`）：

```bash
curl -F "file=@test.mp3" -F "quality=medium" -F "title=测试" \
  http://127.0.0.1:9000/api/transcribe
```

返回包含：

- `sky_json`：可直接导入 SkyStudio 的 JSON 数组
- `abc15`：ABC1-15 文本谱
- `stats`：音符数、最大同时按键、移调、音域
- `stages`：每个阶段的模式与耗时
- `warnings`：模型回退/降级说明

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/qualities` | 低/中/高三档对应的模型 |
| `POST` | `/api/transcribe` | multipart：`file`、`quality`、`title`、`author` |
| `POST` | `/api/transcribe-url` | JSON：`audio_url`、`quality`、`title`、`author` |

部署到 FC 后，公开地址形如：

```text
https://<fc-function-url>/api/transcribe
```

## Docker

```bash
docker build -t sky-music .
docker run --rm -p 9000:9000 sky-music
```

启用真实模型：

```bash
docker build --build-arg INSTALL_MODELS=1 -t sky-music:models .
docker run --gpus all --rm -p 9000:9000 \
  -e MOCK_MODE=0 \
  -e DEVICE=cuda \
  -e MODEL_DIR=/mnt/models \
  -v /path/to/models:/mnt/models \
  sky-music:models
```

## 模型开关

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MOCK_MODE` | `1` | `0` 才启用真实模型 |
| `DEVICE` | `cuda`/`cpu` | 推理设备 |
| `MODEL_DIR` | `/mnt/models` | 权重目录 |
| `VOCALPARSE_CKPT` | 空 | VocalParse 权重目录 |
| `SUBROFORMER_DIR` | 空 | SubRoformer checkpoint |
| `GAME_DIR` | 空 | GAME 仓库目录（含 `infer.py` 和 `models/game-*.pt`） |
| `MAX_DURATION_SEC` | `600` | 单曲最长秒数 |
| `HF_TOKEN` | 空 | HuggingFace gated 权重需要 |

适配器是“能加载就加载，失败就回退并写 warning”，所以镜像本身不会因为缺权重启动失败。

## 阿里云 FC

镜像构建完成后地址为：

```text
ghcr.io/<你的GitHub用户名>/sky-music:latest
```

FC 创建 GPU 函数时：

- 卡型：Ada 系列
- 规格：48GB × 2 卡（Ada.2）
- 实例类型：弹性实例
- 最小实例数：0
- 监听端口：9000
- 启动命令：`uvicorn app.main:app --host 0.0.0.0 --port 9000`
- 执行超时：按最长音频设置（例如 3600 秒）
- 临时硬盘：10GB
- 模型：NAS 挂载到 `/mnt/models`，设置 `MOCK_MODE=0`

如果 FC 拉取 ghcr.io 失败，可在 ACR 里做一次镜像同步，或把镜像重新推到 ACR 后使用 ACR 地址。

## 许可提醒

- 代码：本仓库自行编写部分可按 MIT 使用。
- VocalParse：Apache-2.0；SubRoformer / Beat This / All-In-One：MIT。
- GAME 权重：CC BY-NC-SA 4.0；MuScriptor 权重：CC BY-NC 4.0。
- 这些 NC 权重只适用于公益/私人非商业使用；商用需替换或单独授权。
- 用户上传音频的版权由上传者负责；建议只处理本人有权使用的内容，并设置短期删除策略。
