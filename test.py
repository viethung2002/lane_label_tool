import json
import numpy as np
import os

def generate_points_from_line(line):
    x1, y1 = line[0]
    x2, y2 = line[1]
    points = []
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    intercept = y1 - slope * x1
    y_start = round(y1 / 10) * 10
    y_end = round(y2 / 10) * 10
    if y_start < y_end:
        y_range = np.arange(y_start, y_end + 10, 10)
    else:
        y_range = np.arange(y_start, y_end - 10, -10)
    for y in y_range:
        if slope != float('inf'):
            x = (y - intercept) / slope
        else:
            x = x1
        points.append((x, y))
    return points

def generate_points_from_curve(points):
    all_points = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        intercept = y1 - slope * x1
        y_start = round(y1 / 10) * 10
        y_end = round(y2 / 10) * 10
        if y_start < y_end:
            y_range = np.arange(y_start, y_end + 10, 10)
        else:
            y_range = np.arange(y_start, y_end - 10, -10)
        for y in y_range:
            if slope != float('inf'):
                x = (y - intercept) / slope
            else:
                x = x1
            all_points.append((x, y))
    # Thêm điểm cuối nếu chưa có
    if points[-1] not in all_points:
        all_points.append(points[-1])
    return all_points

# Đường dẫn file JSON (thay bằng đường dẫn thực tế của bạn)
json_file_path = r"D:\lane_labeling_tool\annotations\0003.json"  # Thay bằng đường dẫn thực tế
output_txt_path = "gt_image/output.lines.txt"  # Đường dẫn file txt đầu ra

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

# Đọc file JSON
try:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File {json_file_path} not found.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: File {json_file_path} is not a valid JSON.")
    exit(1)

# Sinh điểm từ dữ liệu JSON
points_per_object = []

for obj in data["annotations"]:
    obj_points = []
    for annotation in obj["annotations"]:
        if annotation[0] == "line":
            line = [annotation[1], annotation[2]]
            points = generate_points_from_line(line)
            obj_points.extend(points)
        elif annotation[0] == "curve":
            points = generate_points_from_curve(annotation[1])
            obj_points.extend(points)
    
    if obj_points:
        # Sắp xếp theo Y giảm dần
        sorted_points = sorted(obj_points, key=lambda point: point[1], reverse=True)
        points_per_object.append(sorted_points)

# Ghi điểm vào file txt
try:
    with open(output_txt_path, "w") as file:
        for obj_points in points_per_object:
            line_str = " ".join([f"{point[0]:.4f} {point[1]}" for point in obj_points])
            file.write(line_str + "\n")
    print(f"Points saved to {output_txt_path}")
except Exception as e:
    print(f"Error saving to file: {str(e)}")

# (Tùy chọn) In ra console để kiểm tra
for i, obj_points in enumerate(points_per_object):
    line_str = " ".join([f"{point[0]:.4f} {point[1]}" for point in obj_points])
    print(f"Lane {data['annotations'][i]['id']}:")
    print(line_str)
    print()
