# Use hardened pytorch image as starting point.
FROM dhi.io/pytorch:2.11.0-cuda12.8-cudnn9

WORKDIR /app

COPY requirements.txt .
RUN ["pip", "install", "--no-cache-dir", "-r", "requirements.txt"]
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

COPY server/server.py .
COPY server/index.html .
# TODO: copy fine-tuned model

# Set 8000 as container listening port.
EXPOSE 8000

CMD ["/home/nonroot/.local/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]