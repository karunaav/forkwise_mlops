# ============================================================
# ForkWise Training Container — NVIDIA RTX 6000 (CUDA)
# ============================================================
# Base: PyTorch 2.3.1 with CUDA 12.1 and cuDNN 9
#
# Build:
#   docker build -t subst-training:latest -f training/docker_nvidia/Dockerfile .
#
# Run (GPU):
#   docker run --rm --gpus all \
#     -v $(pwd):/workspace \
#     --shm-size=12g --network host \
#     subst-training:latest \
#     python training/train.py --config training/config.yaml \
#       --dataset /workspace/data/processed/train.json \
#       --run_name gismo-emb128-gpu \
#       --mlflow_tracking_uri http://<HOST_IP>:5000
#
# Run (CPU only, for baseline):
#   docker run --rm \
#     -v $(pwd):/workspace \
#     --shm-size=12g --network host \
#     subst-training:latest \
#     python training/train.py --config training/config.yaml \
#       --dataset /workspace/data/processed/train.json \
#       --run_name baseline-cpu \
#       --mlflow_tracking_uri http://<HOST_IP>:5000
# ============================================================

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.4.2 \
    "pandas>=2.2.2" \
    tqdm==4.66.2 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    boto3>=1.34.0 \
    setuptools

# Copy entire project (training reads from /workspace/training/ and /workspace/data/)
COPY . /workspace

# training/ directory contains train.py, evaluate.py, model_stub.py
ENV PYTHONPATH=/workspace/training
ENV PYTHONUNBUFFERED=1

CMD ["python", "training/train.py", "--config", "training/config.yaml"]
