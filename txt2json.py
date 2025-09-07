
import os
import cv2
import json
import random  # Thêm để tạo màu ngẫu nhiên

folder_path = r"D:\lane_labeling_tool\gt_image"  # Thư mục chứa ảnh + txt
save_folder = r"D:\lane_labeling_tool\annotations"  # Thư mục lưu JSON

# Tạo thư mục lưu JSON nếu chưa tồn tại
os.makedirs(save_folder, exist_ok=True)

# Tìm tất cả file ảnh
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

for img_file in image_files:
    txt_file = img_file.replace(".jpg", ".lines.txt")
    img_path = os.path.join(folder_path, img_file)
    txt_path = os.path.join(folder_path, txt_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh {img_path}, bỏ qua.")
        continue

    annotations = []
    lane_id = 1

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                nums = list(map(float, line.strip().split()))
                if len(nums) < 4:
                    continue  # Bỏ qua nếu không đủ điểm

                # Chuyển thành polyline từ các điểm
                points = [(int(nums[i]), int(nums[i + 1])) for i in range(0, len(nums), 2)]
                polyline_segments = []
                lane_color = [random.randint(0, 255) for _ in range(3)]  # Màu ngẫu nhiên

                for i in range(len(points) - 1):
                    seg = {
                        "start": [points[i][0], points[i][1]],
                        "end": [points[i + 1][0], points[i + 1][1]],
                        "color": lane_color,
                        "thickness": 2
                    }
                    polyline_segments.append(seg)

                lane_annotation = {
                    "id": lane_id,
                    "annotations": [["polyline", polyline_segments]],
                    "gray_value": 0  # Có thể thay đổi nếu muốn
                }
                annotations.append(lane_annotation)
                lane_id += 1

    # Tạo JSON theo format
    json_data = {
        "annotations": annotations,
        "lane_characteristics": ["Normal"]
    }

    # Lưu JSON vào thư mục save_folder
    json_name = img_file.replace(".jpg", ".json")
    json_path = os.path.join(save_folder, json_name)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, indent=4)

    print(f"Đã tạo {json_path}")
