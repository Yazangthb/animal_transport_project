from api.inference_reasoning import get_reasoning_wrapper

reasoning = get_reasoning_wrapper()

# Sample input features (as if from VLM + distance calculation)
sample_features = {
    "animal_category": "bird",
    "size_class": "medium",
    "is_domesticated": False,
    "dangerous_to_humans": False,
    "distance_km": 3200
}

print("Testing reasoning LLM with sample input:")
print(sample_features)
print("\nResponse:")
result = reasoning.plan_transport(sample_features)
print(result)