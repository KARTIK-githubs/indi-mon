import tensorflow as tf
import pickle
import os

MODEL_PATH = "indimon_monument_model.h5"
PKL_PATH = "indimon_monument_model.pkl"
TRAIN_PATH = r"C:\Users\karti\Downloads\archive (2)\Indian-monuments\images\train"

# ===== Load model =====
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Get class names =====
class_names = sorted(
    [folder for folder in os.listdir(TRAIN_PATH)
     if os.path.isdir(os.path.join(TRAIN_PATH, folder))]
)

# ===== Create wrapper dict =====
model_package = {
    "model": model,
    "class_names": class_names,
    "img_size": (224, 224)
}

# ===== Save as PKL =====
with open(PKL_PATH, "wb") as f:
    pickle.dump(model_package, f)

print("✅ Model exported as PKL:", PKL_PATH)