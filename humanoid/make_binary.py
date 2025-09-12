import os
import cv2

dir = "humanoid_data_images"

for subdir in sorted(os.listdir(dir)):
    subdir_path = os.path.join(dir, subdir)
    if not os.path.isdir(subdir_path):
        continue
    for fname in sorted(os.listdir(subdir_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(subdir_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load {img_path}")
            continue
        # Apply binary threshold
        _, img_binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite(img_path, img_binary)

print("All images converted to binary masks.")