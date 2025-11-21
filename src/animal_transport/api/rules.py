from typing import List, Dict

TRANSPORT_MODES = [
    "truck",
    "train",
    "ship",
    "passenger_plane_cabin",
    "passenger_plane_cargo",
    "cargo_plane",
]

CATEGORY_ALLOWED_MODES: Dict[str, List[str]] = {
    "small_pet": ["truck", "train", "passenger_plane_cabin", "passenger_plane_cargo"],
    "medium_pet": ["truck", "train", "passenger_plane_cargo", "cargo_plane"],
    "large_livestock": ["truck", "ship", "cargo_plane"],
    "bird": ["truck", "ship", "passenger_plane_cargo", "cargo_plane"],
    "reptile": ["truck", "passenger_plane_cargo", "cargo_plane"],
    "wild_dangerous": ["truck", "cargo_plane"],
    "other_unknown": ["truck", "cargo_plane"],
}

MODE_SPEED_KMH: Dict[str, float] = {
    "truck": 60.0,
    "train": 90.0,
    "ship": 25.0,
    "passenger_plane_cabin": 800.0,
    "passenger_plane_cargo": 750.0,
    "cargo_plane": 700.0,
}

MODE_OVERHEAD_HOURS: Dict[str, float] = {
    "truck": 0.5,
    "train": 0.25,
    "ship": 2.0,
    "passenger_plane_cabin": 1.0,
    "passenger_plane_cargo": 1.5,
    "cargo_plane": 2.0,
}


def compute_travel_time_hours(mode: str, distance_km: float) -> float:
    speed = MODE_SPEED_KMH[mode]
    overhead = MODE_OVERHEAD_HOURS[mode]
    return distance_km / speed + overhead
