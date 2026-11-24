import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).with_name("indimon_monument_model.h5")
IMG_SIZE = (224, 224)

# Class order stored with the original model export.
class_names = (
    "Ajanta Caves",
    "Charar-E- Sharif",
    "Chhota_Imambara",
    "Ellora Caves",
    "Fatehpur Sikri",
    "Gateway of India",
    "Humayun_s Tomb",
    "India gate pics",
    "Khajuraho",
    "Sun Temple Konark",
    "alai_darwaza",
    "alai_minar",
    "basilica_of_bom_jesus",
    "charminar",
    "golden temple",
    "hawa mahal pics",
    "iron_pillar",
    "jamali_kamali_tomb",
    "lotus_temple",
    "mysore_palace",
    "not_monument",
    "qutub_minar",
    "tajmahal",
    "tanjavur temple",
    "victoria memorial",
)


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

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    predictions = model.predict(img_array, verbose=0)[0]
    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    monument_name = class_names[class_index]

    print("\n🔍 Prediction Result")

    if monument_name == "not_monument":
        print("Result: Not a monument")
        print(f"Confidence: {confidence*100:.2f}%")

    elif confidence < CONF_THRESHOLD_UNKNOWN:
        print("Result: Unknown (low classifier confidence)")
        print(f"Low confidence: {confidence*100:.2f}%")

    elif confidence < CONF_THRESHOLD_MONUMENT:
        print(f"Result: Uncertain monument ({monument_name})")
        print(f"Confidence: {confidence*100:.2f}%")

    else:
        print(f"Monument: {monument_name}")
        print(f"Confidence: {confidence*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a photo with the bundled INDIMON model.")
    parser.add_argument("image", type=Path, help="Path to a JPEG or PNG photo")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")
    predict_image(args.image)
