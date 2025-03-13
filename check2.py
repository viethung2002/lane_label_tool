import cv2

# Đọc ảnh gốc
image = cv2.imread("D:/lane_labeling_tool/gt_image/0008.png")

# Đọc các điểm từ file .txt
points = []
with open("D:/lane_labeling_tool/gt_image/0008_lines.txt", "r") as file:
    for line in file:
        # Tách các giá trị x, y trong mỗi dòng
        values = line.split()
        for i in range(0, len(values), 2):
            x = float(values[i])
            y = float(values[i + 1])
            points.append((int(x), int(y)))

# Vẽ các điểm lên ảnh
for point in points:
    cv2.circle(image, point, 2, (0, 0, 255), -1)  # Vẽ điểm với màu đỏ

# Lưu ảnh đã vẽ
output_path = "D:/lane_labeling_tool/image_with_points.png"
cv2.imwrite(output_path, image)

# Hiển thị ảnh kết quả
output_path
