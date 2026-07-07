# Use hardened pytorch image as starting point.
FROM dhi.io/pytorch:2.11.0-cuda12.8-cudnn9

WORKDIR /app

COPY requirements.txt .
RUN ["pip", "install", "--no-cache-dir", "-r", "requirements.txt"]

COPY server/server.py .
COPY server/index.html .
COPY server/fuzzy_match.py .
COPY server/drug_dictionary.csv .
COPY server/rx-donut-final/ ./rx-donut-final/

# Set 8000 as container listening port.
EXPOSE 8000

CMD ["python", "server.py"]
# CMD ["/bin/sh", "-c", "/home/nonroot/.local/bin/uvicorn server:app --host 0.0.0.0 --port $PORT"]