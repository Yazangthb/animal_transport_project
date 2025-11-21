import base64
import sys

sys.path.insert(0, 'src')
from animal_transport.api.inference_vlm import get_vlm_wrapper

vlm = get_vlm_wrapper()

with open("tests/tiger.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

print(vlm.analyze_animal(img_base64))
