import json
import random
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from animal_transport.api.rules import (
    CATEGORY_ALLOWED_MODES, 
    TRANSPORT_MODES, 
    compute_travel_time_hours,
    get_size_class,
    is_domesticated,
    is_dangerous_to_humans,
    compute_allowed_and_disallowed_modes,
    generate_notes_text,
    generate_reasoning_text
)
from animal_transport.api.prompts import SYSTEM_PROMPT

OUTPUT_PATH = Path("data/train/train.jsonl")

ANIMAL_CATEGORIES = list(CATEGORY_ALLOWED_MODES.keys())


def sample_example() -> dict:
    # Use deterministic rules instead of random assignment
    animal_category = random.choice(ANIMAL_CATEGORIES)
    size_class = get_size_class(animal_category)
    domesticated_status = is_domesticated(animal_category)
    dangerous_status = is_dangerous_to_humans(animal_category, domesticated_status)
    
    # Random distance but deterministic attribute computation
    distance_km = round(random.uniform(50, 10000), 2)

    # Use MUTUAL FUNCTION for consistent rule computation with evaluator
    allowed, disallowed = compute_allowed_and_disallowed_modes(
        animal_category, size_class, domesticated_status, dangerous_status, distance_km
    )

    # Generate available modes with deterministic time computation
    available_modes = []
    for mode in allowed:
        t = compute_travel_time_hours(mode, distance_km)
        available_modes.append(
            {
                "mode": mode,
                "estimated_time_hours": round(t, 2),
                "notes": generate_notes_text(mode, animal_category, distance_km),
            }
        )

    # Use MUTUAL FUNCTION for consistent reasoning generation with evaluator
    reasoning = generate_reasoning_text(
        animal_category, size_class, domesticated_status, dangerous_status, distance_km
    )

    # Input features as specified in SYSTEM_PROMPT contract
    input_features = {
        "animal_category": animal_category,
        "size_class": size_class,
        "is_domesticated": domesticated_status,
        "dangerous_to_humans": dangerous_status,
        "distance_km": distance_km,
    }

    # Output JSON as specified in SYSTEM_PROMPT contract
    output_json = {
        "available_modes": available_modes,
        "disallowed_modes": disallowed,
        "reasoning": reasoning,
    }

    # Chat message format exactly as specified
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(input_features, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": json.dumps(output_json, ensure_ascii=False),
        },
    ]

    return {"messages": messages}


def main(num_samples: int = 2000):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for _ in range(num_samples):
            example = sample_example()
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {num_samples} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
