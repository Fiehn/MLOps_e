FROM trainer:latest

RUN mkdir /reports && mkdir /models

WORKDIR /
RUN pip install uv
RUN uv sync --no-cache

ENTRYPOINT ["uv","run", "mlops/train.py"]