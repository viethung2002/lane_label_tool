import os
from PIL import Image

input_folder = 'output_frames'
output_dir = r"D:\lane_labeling_tool\resize_image"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Mở từ input_folder
        image = Image.open(os.path.join(input_folder, filename))

        # Resize
        image_resize = image.resize((1640, 590))

        # Lưu vào output_dir
        image_resize.save(os.path.join(output_dir, filename))
