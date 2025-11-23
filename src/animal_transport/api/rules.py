from typing import List, Dict, Tuple

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

# Deterministic attribute rules for dataset generation
# These rules ensure zero logical drift between generator and evaluator

def get_size_class(animal_category: str) -> str:
    """
    Deterministic size class assignment based on animal category.
    
    Args:
        animal_category: The animal category
        
    Returns:
        The appropriate size class string
    """
    size_mapping = {
        "small_pet": "small",
        "medium_pet": "medium", 
        "large_livestock": "large",
        "bird": "small",
        "reptile": "small",
        "wild_dangerous": "large",
        "other_unknown": "medium",
    }
    return size_mapping.get(animal_category, "medium")

def is_domesticated(animal_category: str) -> bool:
    """
    Deterministic domestication status based on animal category.
    
    Args:
        animal_category: The animal category
        
    Returns:
        True if the animal is typically domesticated, False otherwise
    """
    domesticated_categories = {
        "small_pet": True,
        "medium_pet": True, 
        "large_livestock": True,
        "bird": True,  # Most birds in transport contexts are domesticated
        "reptile": False,  # Most reptiles are wild or exotic
        "wild_dangerous": False,
        "other_unknown": False,  # Default to wild for unknown categories
    }
    return domesticated_categories.get(animal_category, False)

def is_dangerous_to_humans(animal_category: str, is_domesticated: bool) -> bool:
    """
    Deterministic dangerous status based on animal category and domestication.
    
    Args:
        animal_category: The animal category
        is_domesticated: Whether the animal is domesticated
        
    Returns:
        True if the animal is dangerous to humans, False otherwise
    """
    # Always dangerous categories
    if animal_category == "wild_dangerous":
        return True
    
    # Reptiles are often dangerous regardless of domestication
    if animal_category == "reptile":
        return True
        
    # Wild birds can be dangerous
    if animal_category == "bird" and not is_domesticated:
        return True
        
    # Otherwise, assume not dangerous
    return False

def compute_travel_time_hours(mode: str, distance_km: float) -> float:
    speed = MODE_SPEED_KMH[mode]
    overhead = MODE_OVERHEAD_HOURS[mode]
    return distance_km / speed + overhead


def compute_allowed_and_disallowed_modes(
    animal_category: str,
    size_class: str,
    is_domesticated: bool,
    dangerous_to_humans: bool,
    distance_km: float
) -> Tuple[List[str], List[str]]:
    """
    MUTUAL FUNCTION: Computes allowed and disallowed transport modes using the exact same logic
    as the rule evaluator. This function ensures zero logical drift between generator and evaluator.
    
    Args:
        animal_category: The animal category
        size_class: The size class
        is_domesticated: Whether the animal is domesticated
        dangerous_to_humans: Whether the animal is dangerous to humans
        distance_km: The transport distance in kilometers
        
    Returns:
        Tuple of (allowed_modes, disallowed_modes) both as sorted lists
    """
    # Base allowed modes from category
    base_allowed = set(CATEGORY_ALLOWED_MODES[animal_category])
    disallowed = [m for m in TRANSPORT_MODES if m not in base_allowed]

    # Apply additional constraints
    # Non-domesticated animals cannot use passenger cabin
    if not is_domesticated and "passenger_plane_cabin" in base_allowed:
        base_allowed.remove("passenger_plane_cabin")
        if "passenger_plane_cabin" not in disallowed:
            disallowed.append("passenger_plane_cabin")

    # Sort for deterministic output
    allowed = sorted(base_allowed)
    disallowed = sorted(list(set(disallowed)))
    
    return allowed, disallowed


def generate_notes_text(mode: str, animal_category: str, distance_km: float) -> str:
    """
    MUTUAL FUNCTION: Generates notes in the exact format expected by the evaluator.
    
    Args:
        mode: The transport mode
        animal_category: The animal category
        distance_km: The transport distance
        
    Returns:
        Notes string in exact evaluator format
    """
    return f"Mode {mode} for {animal_category} over {distance_km} km."


def generate_reasoning_text(
    animal_category: str,
    size_class: str,
    is_domesticated: bool,
    dangerous_to_humans: bool,
    distance_km: float
) -> str:
    """
    MUTUAL FUNCTION: Generates reasoning text in the exact format expected by the evaluator.
    
    Args:
        animal_category: The animal category
        size_class: The size class
        is_domesticated: Whether the animal is domesticated
        dangerous_to_humans: Whether the animal is dangerous to humans
        distance_km: The transport distance
        
    Returns:
        Reasoning string in exact evaluator format
    """
    return (
        f"Animal category {animal_category}, size {size_class}, "
        f"{'domesticated' if is_domesticated else 'not domesticated'}, "
        f"{'dangerous' if dangerous_to_humans else 'not dangerous'}, "
        f"distance {distance_km} km. Allowed modes chosen based on predefined rules."
    )
