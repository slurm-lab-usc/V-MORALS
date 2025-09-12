import os
import cv2

traj_dir = "pendulum_data_images"
output_dir = "pendulum_data_images_128"
os.makedirs(output_dir, exist_ok=True)

for subdir in sorted(os.listdir(traj_dir)):
    input_subdir = os.path.join(traj_dir, subdir)
    output_subdir = os.path.join(output_dir, subdir)
    if not os.path.isdir(input_subdir):
        continue
    os.makedirs(output_subdir, exist_ok=True)

    for fname in sorted(os.listdir(input_subdir)):
        if not fname.endswith(".png"):
            continue
        img_path = os.path.join(input_subdir, fname)
        out_path = os.path.join(output_subdir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}")
            continue
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, resized)

print("All images resized to 128x128 and saved to", output_dir)