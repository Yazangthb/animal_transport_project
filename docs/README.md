# Animal Transport Reasoning System

This project implements a multimodal research system that determines **allowed transportation modes** and **estimated travel times** for an animal, given:

- an **image of the animal**
- **start coordinates** (lat/lon)
- **end coordinates**

The system uses:

- **Qwen/Qwen2.5-VL-7B-Instruct** (Vision-Language Model)
- **Qwen/Qwen2.5-7B-Instruct** fine-tuned using **QLoRA**
- **Synthetic dataset generation**
- **FastAPI** for inference
- **Docker** with **GPU support**
- Fully local inference (no external APIs)

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Python 3.10+ (for local development)

### Using Docker (Recommended)
```bash
# Build and run training
docker build -f docker/Dockerfile.training -t animal-train .
docker run --gpus all -it animal-train

# Inside container, generate dataset and train
python scripts/generate_dataset.py
python scripts/train.py

# Build and run API
docker build -f docker/Dockerfile.api -t animal-api .
docker run --gpus all -p 8000:8000 animal-api
```

### Local Development
```bash
# Setup environment
conda create -n animal python=3.10 -y
conda activate animal
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


# Generate dataset
python scripts/generate_dataset.py

# Train model
python scripts/train.py

# Run API
uvicorn src.animal_transport.api.main:app --host 0.0.0.0 --port 8000
```

---

# 📦 Installation Guide

Choose between Docker (recommended for production) or local development setup.

## 🔧 Local Development Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.1+
- Conda (recommended for environment management)

### 1. Create Python Environment
```bash
conda create -n animal python=3.10 -y
conda activate animal
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🐳 Docker Setup (Production)

### Prerequisites
- NVIDIA GPU with working driver
- Docker with NVIDIA Container Toolkit

---

# 1️⃣ Install NVIDIA GPU Driver

Your host machine **must** have a working NVIDIA driver.  
Check:

```
nvidia-smi
```

If this fails, install:

```
sudo ubuntu-drivers autoinstall
reboot
```

---

# 2️⃣ Install Docker (Official Version)

Remove old installations:

```
sudo apt remove docker docker.io containerd runc
sudo snap remove docker 2>/dev/null
```

Install dependencies:

```
sudo apt update
sudo apt install ca-certificates curl gnupg
```

Add Docker GPG key:

```
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

Add Docker repo:

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
```

Install Docker:

```
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Enable non-root access:

```
sudo usermod -aG docker $USER
newgrp docker
```

Test:

```
docker run hello-world
```

---

# 3️⃣ Install NVIDIA Container Toolkit

Add toolkit repo:

```
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu20.04/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
```

Install:

```
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

Configure Docker:

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

# 4️⃣ Test GPU Inside Docker

```
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If GPU info appears, you're good.

---

# 5️⃣ Python Environment (Local)

Create env:

```
conda create -n animal python=3.10 -y
conda activate animal
```

Install deps:

```
pip install -r docker/requirements.txt
```

Freeze pip:

```
pip freeze > docker/requirements.txt
```

Freeze conda:

```
conda env export --no-builds > environment.yml
```

---

# 6️⃣ Docker Images (Training + API)

Build training image:

```
docker build -f docker/Dockerfile.training -t animal-train .
```

Train:

```
docker run --gpus all -it animal-train bash
python scripts/train.py
```

Build API image:

```
docker build -f docker/Dockerfile.api -t animal-api .
```

Run API:

```
docker run --gpus all -p 8000:8000 animal-api
```

---

# 🚀 Running the Project

## Dataset Generation
Generate synthetic training data:
```bash
# Local
python scripts/generate_dataset.py

# Docker
docker run --gpus all -it animal-train python scripts/generate_dataset.py
```

## Model Training
Fine-tune the reasoning model:
```bash
# Local
python scripts/train.py

# Docker
docker run --gpus all -it animal-train python scripts/train.py
```

## API Server
Start the FastAPI inference server:
```bash
# Local
uvicorn src.animal_transport.api.main:app --host 0.0.0.0 --port 8000

# Docker
docker run --gpus all -p 8000:8000 animal-api
```

## Testing
Run the test suite:
```bash
# Local
python -m pytest tests/

# Or run individual tests
python tests/test_LLM_inference.py
python tests/test_VLM_inference.py
```

## API Usage
Once the server is running, you can make requests:

```python
import requests
import base64

# Prepare image
with open("path/to/animal.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/infer",
    json={
        "file": img_data,  # base64 encoded image
        "start_lat": 48.85,
        "start_lon": 2.35,
        "end_lat": 51.5074,
        "end_lon": -0.1278
    }
)

print(response.json())
```

---

# 🧠 System Overview

```
User Input (image + coords)
    │
    ▼
Qwen2.5-VL-7B (VLM)
→ Animal classification JSON
    │
    ▼
Haversine distance
    │
    ▼
Qwen2.5-7B + LoRA
→ Allowed modes
→ Time estimates
→ Reasoning
```


---

# 🛠️ Development

## Code Structure
The project follows a modular architecture:

- **`src/animal_transport/`**: Main package
  - **`api/`**: FastAPI application and inference logic
  - **`train/`**: Training pipeline components
- **`scripts/`**: Executable scripts for training and data generation
- **`tests/`**: Unit and integration tests
- **`data/`**: Datasets and training data
- **`models/`**: Saved model checkpoints
- **`docker/`**: Container configurations

## Adding New Features
1. For API features: Add to `src/animal_transport/api/`
2. For training features: Add to `src/animal_transport/train/`
3. For scripts: Add to `scripts/`
4. Update imports and tests accordingly

## Environment Variables
Create a `.env` file in the project root:
```bash
REASONING_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
VLM_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
```

---

# 📁 Project Structure

```
animal_transport_project/
├── src/
│   └── animal_transport/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py          # FastAPI app
│       │   ├── config.py        # API configuration
│       │   ├── inference_vlm.py # VLM inference
│       │   ├── inference_reasoning.py # LLM reasoning
│       │   ├── schemas.py       # Pydantic models
│       │   ├── prompts.py       # System prompts
│       │   ├── rules.py         # Transport rules
│       │   └── utils_distance.py # Distance calculations
│       └── train/
│           ├── __init__.py
│           ├── config.py        # Training config
│           ├── data.py          # Dataset class
│           ├── model.py         # Model loading & LoRA
│           └── train.py         # Training logic
├── scripts/
│   ├── train.py                 # Training entry point
│   └── generate_dataset.py      # Dataset generation
├── data/
│   └── train/
│       └── train.jsonl          # Training dataset
├── models/                      # Saved models
├── tests/
│   ├── test_LLM_inference.py
│   ├── test_VLM_inference.py
│   ├── test_qwen.py
│   └── tiger.jpg                # Test image
├── docker/
│   ├── Dockerfile.training
│   ├── Dockerfile.api
│   ├── requirements.txt
│   └── environment.yml
├── docs/
│   └── README.md
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
├── .env                         # Environment variables
├── .env.example                 # Example env file
└── .gitignore
```

