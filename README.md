
# Animal Transport Reasoning System

This project implements a multimodal research system that determines allowed transportation modes and estimated travel times for an animal, given:

- an **image of the animal**
- **starting coordinates** (lat/lon)
- **ending coordinates**

The system uses:

- **Qwen/Qwen2.5-VL-7B-Instruct** as a Vision-Language Model (VLM)
- **Qwen/Qwen2.5-7B-Instruct** (or any ≤7B LLM) fine‑tuned with **QLoRA**
- **Synthetic dataset generation**
- **FastAPI** for inference
- **Docker** for packaging
- **No external APIs** (fully self-contained)

---

# 🔧 Features

### ✔ Image → Animal category via VLM  
The VLM extracts structured info:
```json
{
  "animal_category": "bird",
  "size_class": "medium",
  "is_domesticated": false,
  "dangerous_to_humans": false
}
```

### ✔ Deterministic distance estimation  
Using Haversine formula.

### ✔ LLM reasoning  
The fine‑tuned LLM determines:

- **Allowed transport modes**
- **Disallowed modes**
- **Travel time estimates**
- **Short reasoning**

### ✔ FastAPI endpoint  
`POST /infer` returns a full structured response.

### ✔ Docker images  
One for training, one for inference.

---

# 📂 Directory Structure

```
animal_transport_project/
│
├── README.md
├── report.pdf                
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.training
│   └── requirements.txt
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── utils_distance.py
│   ├── rules.py
│   ├── inference_vlm.py
│   ├── inference_reasoning.py
│   └── config.py
│
├── train/
│   ├── generate_dataset.py
│   ├── finetune_llm.py
│   └── dataset/
│       └── train.jsonl
│
└── models/
    ├── reasoning_lora/       # Output of fine-tuning
    ├── base_vlm/             # local cache
    └── base_reasoning/
```

---

# 🚀 System Architecture

```
User Input (image + coords)
           │
           ▼
Qwen2.5 VL-7B (VLM)
→ animal classification
           │
           ▼
Distance calculator (Haversine)
           │
           ▼
Reasoning LLM (fine‑tuned)
→ available modes
→ time estimates
           │
           ▼
FastAPI response (JSON)
```

---

# 🔍 VLM Output Schema

The VLM must output JSON with:

```json
{
  "animal_category": "small_pet | medium_pet | large_livestock | bird | reptile | wild_dangerous | other_unknown",
  "size_class": "small | medium | large",
  "is_domesticated": true,
  "dangerous_to_humans": false
}
```

These features drive the reasoning model.

---

# 📐 Reasoning Input Schema

```json
{
  "animal_category": "bird",
  "size_class": "medium",
  "is_domesticated": false,
  "dangerous_to_humans": false,
  "distance_km": 3200
}
```

---

# 🧠 Reasoning Output Schema

```json
{
  "available_modes": [
    {
      "mode": "cargo_plane",
      "estimated_time_hours": 6.5,
      "notes": "Recommended for long distances."
    }
  ],
  "disallowed_modes": ["train"],
  "reasoning": "..."
}
```

---

# 📊 Transport Rules

These rules define your ground truth for synthetic dataset generation and LLM fine‑tuning.

### Allowed modes by category:

| Category          | Modes |
|------------------|------------------------------------------------------|
| small_pet        | truck, train, passenger_plane_cabin, passenger_plane_cargo |
| medium_pet       | truck, train, passenger_plane_cargo, cargo_plane |
| large_livestock  | truck, ship, cargo_plane |
| bird             | truck, ship, passenger_plane_cargo, cargo_plane |
| reptile          | truck, passenger_plane_cargo, cargo_plane |
| wild_dangerous   | truck, cargo_plane |
| other_unknown    | truck, cargo_plane |

### Transport speeds (km/h)

```
truck = 60
train = 90
ship = 25
passenger_plane_cabin = 800
passenger_plane_cargo = 750
cargo_plane = 700
```

### Overheads (hours)

```
truck = 0.5
train = 0.25
ship = 2
passenger_plane_cabin = 1
passenger_plane_cargo = 1.5
cargo_plane = 2
```

### Time estimate formula

```
time = distance_km / speed + overhead
```

---

# 🧪 Dataset Generation

Run:

```
python train/generate_dataset.py
```

Default size: **5000 samples**.

## Configuration

Model names and paths can be configured via environment variables or `.env` file. Copy `.env.example` to `.env` and modify as needed.

Defaults:
- VLM: Qwen/Qwen2.5-VL-7B-Instruct
- Reasoning: Qwen/Qwen2.5-7B-Instruct

Each example contains:

### Input (VLM → LLM):

```json
{
  "animal_category": "reptile",
  "size_class": "small",
  "is_domesticated": false,
  "dangerous_to_humans": true,
  "distance_km": 1200
}
```

### Output:

```json
{
  "available_modes": [...],
  "disallowed_modes": [...],
  "reasoning": "..."
}
```

Saved in ChatML format for LLM training.

---

# 🏋️ Finetuning the Reasoning Model (QLoRA)

Run:

```
python train/finetune_llm.py
```

This:

- loads the base LLM
- applies LoRA adapters
- trains on the synthetic dataset
- saves the adapter to `models/reasoning_lora/`

Hardware:

- 24GB GPU (e.g., RTX 3090)  
- or any A100  
- Training time: ~45–90 mins

---

# ⚙️ FastAPI Service

Start server with Docker:

```
docker build -f docker/Dockerfile.api -t animal-transport-api .
docker run --gpus all -p 8000:8000 animal-transport-api
```

Then test:

```
curl -X POST http://localhost:8000/infer   -H "Content-Type: application/json"   -d '{
    "image_base64": "<your-base64-image>",
    "start": {"lat": 48.85, "lon": 2.35},
    "end": {"lat": 40.71, "lon": -74.00}
  }'
```

Response:

```json
{
  "animal": {...},
  "distance_km": 5837.2,
  "transport": {
    "available_modes": [...],
    "disallowed_modes": [...],
    "reasoning": "..."
  }
}
```

---

# 🐳 Docker Overview

### `Dockerfile.api`

- Runs FastAPI server
- Loads VLM + Reasoning LLM + LoRA
- GPU accelerated

### `Dockerfile.training`

- Generates dataset
- Runs QLoRA fine‑tuning
