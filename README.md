# AIG 200 Capstone

## REST API Server

### Requirements
See: `src/requirements.txt`. <br>
In addition to these libraries PyTorch is required. CUDA-enabled PyTorch is recommended.<br>
`pip install torch torchaudio torchvision`<br>
OR <br>
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

### Run

#### Local
1. Install dependencies above.
2. Command: `uvicorn src.server.server:app --port 8000`
3. Open browser and navigate to http://127.0.0.1/8000

#### Docker Container
1. Navigate to `src/` directory. 
2. Build: `docker build -t <container-tag> .`
3. Run: `docker run --gpus all -p 8000:8000 [-e HF_TOKEN=<your-hf-token>] <container-tag>`
4. Open browser and navigate to http://127.0.0.1/8000 (localhost:8000)