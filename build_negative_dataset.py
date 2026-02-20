import os
import random
import shutil

# =========================
# 🔧 EDIT THESE PATHS
# =========================

HOMEOBJECTS_PATH = r"C:\Users\karti\Downloads\archive (3)\seg_train\seg_train"
INTEL_PATH = r"C:\Users\karti\Downloads\archive (4)\Home_Objects_06\Train"
CIFAR_PATH = r"C:\Users\karti\Downloads\archive (5)\cifar10\train"

OUTPUT_PATH = r"C:\Users\karti\Downloads\archive (2)\Indian-monuments\images\train\not_monument"

TARGET_IMAGES = 2500

# =========================

def collect_images_from_folder(root_folder):
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
    return image_paths


print("🔍 Collecting images...")

all_images = []
all_images += collect_images_from_folder(HOMEOBJECTS_PATH)
all_images += collect_images_from_folder(INTEL_PATH)
all_images += collect_images_from_folder(CIFAR_PATH)

print(f"Total images found: {len(all_images)}")

# random sampling
sampled = random.sample(all_images, min(TARGET_IMAGES, len(all_images)))

# create output folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("📦 Copying sampled images...")

# IMPORTANT: continue numbering if folder already has images
existing_files = len(os.listdir(OUTPUT_PATH))

for i, img_path in enumerate(sampled):
    try:
        new_name = f"neg_{existing_files + i}.jpg"
        shutil.copy(img_path, os.path.join(OUTPUT_PATH, new_name))
    except Exception as e:
        pass

print(f"✅ Done! {len(sampled)} images copied to not_monument")