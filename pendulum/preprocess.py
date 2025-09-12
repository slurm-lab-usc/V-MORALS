import os
import cv2
import numpy as np

input_root = "pendulum_data_images"
output_root = "pendulum_binary"
crop_size = 256 

os.makedirs(output_root, exist_ok=True)

for subdir in sorted(os.listdir(input_root)):
    input_subdir = os.path.join(input_root, subdir)
    output_subdir = os.path.join(output_root, subdir)
    if not os.path.isdir(input_subdir):
        continue
    os.makedirs(output_subdir, exist_ok=True)

    for fname in sorted(os.listdir(input_subdir)):
        if not fname.endswith(".png"):
            continue
        image_path = os.path.join(input_subdir, fname)
        output_path = os.path.join(output_subdir, fname)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load {image_path}")
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 100, 100])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

        # Static crop: center 128x128 region
        h, w = mask.shape
        center_x, center_y = w // 2, h // 2
        x1 = max(center_x - crop_size // 2, 0)
        y1 = max(center_y - crop_size // 2, 0)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Ensure crop does not go out of bounds
        if x2 > w:
            x1 = w - crop_size
            x2 = w
        if y2 > h:
            y1 = h - crop_size
            y2 = h

        cropped_mask = mask[y1:y2, x1:x2]
        cv2.imwrite(output_path, cropped_mask)

print("Batch static cropping complete.")