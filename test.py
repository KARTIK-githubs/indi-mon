import os

path = r"C:\Users\karti\Downloads\archive (2)\Indian-monuments\images\train"

folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
print("Class folders:", len(folders))
print(folders[:10])