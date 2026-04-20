FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends git curl wget vim && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn==1.4.2 pandas>=2.2.2 tqdm==4.66.2 pyyaml==6.0.1 requests==2.31.0 boto3>=1.34.0 setuptools onnx onnxruntime-gpu
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
COPY . /workspace
ENV PYTHONPATH=/workspace/training
ENV PYTHONUNBUFFERED=1
CMD ["python", "training/train.py", "--config", "training/config.yaml"]
