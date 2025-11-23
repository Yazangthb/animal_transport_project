# api/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import base64

from .schemas import InferenceResponse, AnimalInfo, TransportResult, TransportMode
from .utils_distance import haversine_km
from .inference_vlm import get_vlm_wrapper
from .inference_reasoning import get_reasoning_wrapper

app = FastAPI(title="Animal Transport Reasoning Service")

print(f"FastAPI docs enabled at: {app.docs_url}")
print(f"FastAPI redoc enabled at: {app.redoc_url}")

@app.on_event("startup")
async def startup_event():
    print("Application startup complete. Docs should be accessible.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile = File(...),

    # Default: Paris → London
    start_lat: float = Form(48.85),
    start_lon: float = Form(2.35),
    end_lat: float = Form(51.5074),
    end_lon: float = Form(-0.1278),
):
    """
    Supports default coordinates for quick testing:
    - Paris (48.85, 2.35)
    - London (51.5074, -0.1278)

    Users may override any coordinate via form fields.
    """

    try:
        vlm = get_vlm_wrapper()
        reasoning = get_reasoning_wrapper()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")

    # ---------------------------------------------------------------------
    # 1) Read uploaded image → base64
    # ---------------------------------------------------------------------
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # ---------------------------------------------------------------------
    # 2) VLM classification
    # ---------------------------------------------------------------------
    animal_data = vlm.analyze_animal(image_base64)

    animal_info = AnimalInfo(
        animal_category=animal_data["animal_category"],
        size_class=animal_data["size_class"],
        is_domesticated=bool(animal_data["is_domesticated"]),
        dangerous_to_humans=bool(animal_data["dangerous_to_humans"]),
    )

    # ---------------------------------------------------------------------
    # 3) Distance calculation (Haversine)
    # ---------------------------------------------------------------------
    distance_km = haversine_km(start_lat, start_lon, end_lat, end_lon)

    # ---------------------------------------------------------------------
    # 4) Reasoning LLM
    # ---------------------------------------------------------------------
    reasoning_input = {
        "animal_category": animal_info.animal_category,
        "size_class": animal_info.size_class,
        "is_domesticated": animal_info.is_domesticated,
        "dangerous_to_humans": animal_info.dangerous_to_humans,
        "distance_km": distance_km,
    }

    transport_raw = reasoning.plan_transport(reasoning_input)
    print("transport_raw: ", transport_raw)

    available_modes = [
        TransportMode(
            mode=m["mode"],
            estimated_time_hours=float(m["estimated_time_hours"]),
            notes=m.get("notes"),
        )
        for m in transport_raw.get("available_modes", [])
        if isinstance(m, dict) and "mode" in m and "estimated_time_hours" in m
    ]

    transport_result = TransportResult(
        available_modes=available_modes,
        disallowed_modes=transport_raw.get("disallowed_modes", []),
        reasoning=transport_raw.get("reasoning", ""),
    )

    return InferenceResponse(
        animal=animal_info,
        distance_km=distance_km,
        transport=transport_result,
    )
