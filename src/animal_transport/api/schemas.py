from pydantic import BaseModel, Field
from typing import List, Optional


class Point(BaseModel):
    lat: float = Field(..., description="Latitude in degrees")
    lon: float = Field(..., description="Longitude in degrees")


class InferenceRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    start: Point
    end: Point


class AnimalInfo(BaseModel):
    animal_category: str
    size_class: str
    is_domesticated: bool
    dangerous_to_humans: bool


class TransportMode(BaseModel):
    mode: str
    estimated_time_hours: float
    notes: Optional[str] = None


class TransportResult(BaseModel):
    available_modes: List[TransportMode]
    disallowed_modes: List[str]
    reasoning: str


class InferenceResponse(BaseModel):
    animal: AnimalInfo
    distance_km: float
    transport: TransportResult
