import base64

from api.inference_vlm import get_vlm_wrapper

vlm = get_vlm_wrapper()

with open("tiger.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(vlm.analyze_animal(img_base64))
