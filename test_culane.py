import cv2
import numpy as np
import os

folder_path = r"D:\lane_labeling_tool\gt_image"
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
index = 0

def draw_lanes(image, txt_path):
    lanes = []
    if not os.path.exists(txt_path):
        return image
    with open(txt_path, "r") as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            if len(nums) < 2:
                continue  # bỏ qua dòng trống hoặc thiếu điểm
            points = [(int(nums[i]), int(nums[i+1])) for i in range(0, len(nums), 2)]
            if points:  # chỉ thêm nếu có điểm
                lanes.append(points)
    if not lanes:
        return image

    # Sắp xếp theo toạ độ X trung bình, tránh mean của list rỗng
    lanes_sorted = sorted(lanes, key=lambda pts: np.mean([p[0] for p in pts if pts]))

    for idx, lane in enumerate(lanes_sorted, start=1):
        if not lane:
            continue
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        for i in range(len(lane) - 1):
            cv2.line(image, lane[i], lane[i+1], color, 2)
        first_point = lane[0]
        cv2.putText(image, str(idx), (first_point[0] + 5, first_point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image


while True:
    img_path = os.path.join(folder_path, image_files[index])
    txt_path = img_path.replace(".jpg", ".lines.txt")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Lỗi] Không thể đọc ảnh: {img_path}")
        break

    img_with_lanes = draw_lanes(img.copy(), txt_path)
    cv2.putText(img_with_lanes,
                f"{index+1}/{len(image_files)} - {image_files[index]}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Lane Viewer", img_with_lanes)

    key = cv2.waitKeyEx(0)  # Chờ phím bấm
    if key == 27:  # ESC
        break
    elif key in [ord('d'), 2555904]:  # D hoặc mũi tên phải
        index = (index + 1) % len(image_files)
    elif key in [ord('a'), 2424832]:  # A hoặc mũi tên trái
        index = (index - 1) % len(image_files)

cv2.destroyAllWindows()
