import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

# ===== LOAD PKL =====
print("Loading PKL model...")

with open("indimon_monument_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
class_names = data["class_names"]
IMG_SIZE = data["img_size"]

print("✅ Classes loaded:", len(class_names))

# ===== THRESHOLDS =====
CONF_THRESHOLD_MONUMENT = 0.75
CONF_THRESHOLD_UNKNOWN = 0.50


# ===== SAFE IMAGE LOADER =====
def load_and_preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ===== PREDICT FUNCTION =====
def predict_image(img_path):
    img_array = load_and_preprocess(img_path)

    predictions = model.predict(img_array, verbose=0)[0]
    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    monument_name = class_names[class_index]

    print("\n🔍 Prediction Result")

    if monument_name == "not_monument":
        print("Result: Not a monument")
        print(f"Confidence: {confidence*100:.2f}%")

    elif confidence < CONF_THRESHOLD_UNKNOWN:
        print("Result: Suspicious / Possibly AI-generated")
        print(f"Low confidence: {confidence*100:.2f}%")

    elif confidence < CONF_THRESHOLD_MONUMENT:
        print(f"Result: Uncertain monument ({monument_name})")
        print(f"Confidence: {confidence*100:.2f}%")

    else:
        print(f"Monument: {monument_name}")
        print(f"Confidence: {confidence*100:.2f}%")


# ===== TAKE INPUT =====
test_image_path = input("\nEnter image path: ").strip().strip('"')
predict_image(test_image_path)