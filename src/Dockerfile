FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY server/server.py .
COPY server/index.html .
# TODO: copy fine-tuned model

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
# CMD ["sh", "-c", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
# CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
