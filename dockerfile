FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    CONFIG_FILE=/app/conf/conf.yaml

WORKDIR /app

# Base dependencies
RUN apt-get update -o Acquire::Retries=5 \
 && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install deps (cache-friendly)
# [robot fork hook] INSTALL_TORCH=true (the default) builds upstream's image. false leaves out torch and
# its CUDA wheels (~6.6 GB): our compose builds that (VTUBER_INSTALL_TORCH=false in .env), since only
# Silero VAD, FunASR and Coqui TTS import torch and our conf.yaml uses none of them (speech recognition
# is the server's Whisper, see FORK.md). The container starts with `uv run --no-sync`, so uv never
# installs them behind the build's back.
ARG INSTALL_TORCH=true
ARG TORCH_PACKAGES="torch triton cuda-bindings cuda-pathfinder nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12"
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    skip=""; \
    if [ "$INSTALL_TORCH" != "true" ]; then \
      for p in $TORCH_PACKAGES; do skip="$skip --no-install-package $p"; done; \
    fi; \
    uv sync --frozen --no-dev $skip

# [robot fork hook] The AI's tools (mcp_servers.json), pinned and installed as plain commands on PATH, so
# a start downloads nothing (with uvx they were resolved and downloaded again at every start).
ENV UV_TOOL_BIN_DIR=/usr/local/bin
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install mcp-server-time==2026.10.10 \
 && uv tool install duckduckgo-mcp-server==0.7.0

# Copy source & install project
COPY . /app
RUN uv pip install --no-deps .

# Startup script
RUN printf '%s\n' \
  '#!/usr/bin/env sh' \
  'set -eu' \
  '' \
  'mkdir -p /app/conf /app/models' \
  '' \
  '# 1) conf.yaml (required)' \
  'if [ -f "/app/conf/conf.yaml" ]; then' \
  '  echo "Using user-provided conf.yaml"' \
  '  ln -sf /app/conf/conf.yaml /app/conf.yaml' \
  'else' \
  '  echo "ERROR: conf.yaml is required."' \
  '  echo "Please mount your config dir to /app/conf"' \
  '  exit 1' \
  'fi' \
  '' \
  '# 2) model_dict.json (optional)' \
  'if [ -f "/app/conf/model_dict.json" ]; then' \
  '  ln -sf /app/conf/model_dict.json /app/model_dict.json' \
  'fi' \
  '' \
  '# 3) live2d-models' \
  'if [ -d "/app/conf/live2d-models" ]; then' \
  '  rm -rf /app/live2d-models && ln -s /app/conf/live2d-models /app/live2d-models' \
  'fi' \
  '' \
  '# 4) characters' \
  'if [ -d "/app/conf/characters" ]; then' \
  '  rm -rf /app/characters && ln -s /app/conf/characters /app/characters' \
  'fi' \
  '' \
  '# 5) avatars' \
  'if [ -d "/app/conf/avatars" ]; then' \
  '  rm -rf /app/avatars && ln -s /app/conf/avatars /app/avatars' \
  'fi' \
  '' \
  '# 6) backgrounds' \
  'if [ -d "/app/conf/backgrounds" ]; then' \
  '  rm -rf /app/backgrounds && ln -s /app/conf/backgrounds /app/backgrounds' \
  'fi' \
  '' \
  '# 7) start app (--no-sync: keep the environment as built, see INSTALL_TORCH above)' \
  'exec uv run --no-sync run_server.py' \
  > /usr/local/bin/start-app && chmod +x /usr/local/bin/start-app

# Volumes
VOLUME ["/app/conf", "/app/models"]

EXPOSE 12393

CMD ["/usr/local/bin/start-app"]
