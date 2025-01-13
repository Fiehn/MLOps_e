FROM trainer:latest

ENTRYPOINT ["uv","run", "mlops/evaluate.py"]
