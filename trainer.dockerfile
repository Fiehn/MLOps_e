FROM python:3.12-slim
#FROM trainer:latest

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/mlops/ mlops/
COPY data/ data/

RUN mkdir /reports && mkdir /models

WORKDIR /
RUN pip install uv
RUN uv sync --no-cache

ENTRYPOINT ["uv","run", "mlops/train.py"]
