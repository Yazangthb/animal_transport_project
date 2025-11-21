import json
import random
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from animal_transport.api.rules import CATEGORY_ALLOWED_MODES, TRANSPORT_MODES, compute_travel_time_hours
from animal_transport.api.prompts import SYSTEM_PROMPT

OUTPUT_PATH = Path("data/train/train.jsonl")

ANIMAL_CATEGORIES = list(CATEGORY_ALLOWED_MODES.keys())
SIZE_CLASSES = ["small", "medium", "large"]


def sample_example() -> dict:
    animal_category = random.choice(ANIMAL_CATEGORIES)
    if animal_category == "small_pet":
        size_class = "small"
    elif animal_category in ["medium_pet", "reptile", "bird"]:
        size_class = random.choice(["small", "medium"])
    elif animal_category in ["large_livestock", "wild_dangerous"]:
        size_class = random.choice(["medium", "large"])
    else:
        size_class = random.choice(SIZE_CLASSES)

    if animal_category in ["small_pet", "medium_pet", "large_livestock"]:
        is_domesticated = True
    elif animal_category == "wild_dangerous":
        is_domesticated = False
    else:
        is_domesticated = random.choice([True, False])

    if animal_category == "wild_dangerous":
        dangerous_to_humans = True
    elif animal_category == "reptile":
        dangerous_to_humans = random.choice([True, False])
    else:
        dangerous_to_humans = False

    distance_km = round(random.uniform(50, 10000), 2)

    base_allowed = set(CATEGORY_ALLOWED_MODES[animal_category])
    disallowed = [m for m in TRANSPORT_MODES if m not in base_allowed]

    if not is_domesticated and "passenger_plane_cabin" in base_allowed:
        base_allowed.remove("passenger_plane_cabin")
        disallowed.append("passenger_plane_cabin")

    allowed = sorted(base_allowed)
    disallowed = sorted(list(set(disallowed)))

    available_modes = []
    for mode in allowed:
        t = compute_travel_time_hours(mode, distance_km)
        available_modes.append(
            {
                "mode": mode,
                "estimated_time_hours": round(t, 2),
                "notes": f"Mode {mode} for {animal_category} over {distance_km} km.",
            }
        )

    reasoning = (
        f"Animal category {animal_category}, size {size_class}, "
        f"{'domesticated' if is_domesticated else 'not domesticated'}, "
        f"{'dangerous' if dangerous_to_humans else 'not dangerous'}, "
        f"distance {distance_km} km. Allowed modes chosen based on predefined rules."
    )

    input_features = {
        "animal_category": animal_category,
        "size_class": size_class,
        "is_domesticated": is_domesticated,
        "dangerous_to_humans": dangerous_to_humans,
        "distance_km": distance_km,
    }

    output_json = {
        "available_modes": available_modes,
        "disallowed_modes": disallowed,
        "reasoning": reasoning,
    }

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


def main(num_samples: int = 20):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for _ in range(num_samples):
            example = sample_example()
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {num_samples} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
