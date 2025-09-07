import cv2
import numpy as np
import os
from glob import glob

selected_lane_index = -1
lanes = []
labels = []
points_path_global = ""
labels_path_global = ""
image_global = None

def draw_lanes_on_image(image_path, points_path, labels_path):
    global lanes, labels, image_global, points_path_global, labels_path_global
    points_path_global = points_path
    labels_path_global = labels_path

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh từ: " + image_path)

    if not os.path.exists(points_path) or not os.path.exists(labels_path):
        return image  # Trả về ảnh gốc nếu thiếu file

    lanes = []
    with open(points_path, 'r') as f:
        for i, line in enumerate(f, 1):
            values = line.strip().split()
            if not values:
                continue
            if len(values) % 2 != 0:
                print(f"[Cảnh báo] Dòng {i} trong {points_path} có số lượng không chẵn.")
                continue
            points = [(int(float(values[j])), int(float(values[j+1]))) for j in range(0, len(values), 2)]
            lanes.append(points)

    labels = []
    with open(labels_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                labels.append(int(line.strip()))
            except ValueError:
                print(f"[Cảnh báo] Dòng {i} trong {labels_path} không phải số nguyên.")
                continue

    if len(lanes) != len(labels):
        print(f"[Cảnh báo] Số lane ({len(lanes)}) ≠ số label ({len(labels)}).")
        return image

    colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # đỏ và xanh lá

    for i, (lane, label) in enumerate(zip(lanes, labels)):
        color = colors.get(label, (255, 255, 255))
        thickness = 3 if i == selected_lane_index else 1
        for (x, y) in lane:
            cv2.circle(image, (x, y), 5, color, -1)
        if lane:
            x0, y0 = lane[0]
            cv2.putText(image, str(label), (x0 + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if i == selected_lane_index:
                cv2.putText(image, f"SELECTED", (x0 + 40, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    image_global = image.copy()
    return image

def mouse_callback(event, x, y, flags, param):
    global selected_lane_index
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        selected = -1
        for i, lane in enumerate(lanes):
            for px, py in lane:
                dist = np.hypot(px - x, py - y)
                if dist < min_dist and dist < 20:  # chọn lane gần click nhất (trong phạm vi 20px)
                    min_dist = dist
                    selected = i
        selected_lane_index = selected
        if selected >= 0:
            print(f"Chọn lane {selected} (label: {labels[selected]})")
        else:
            print("Không chọn lane nào.")
        redraw_image()

def redraw_image():
    if image_global is not None:
        updated = draw_lanes_on_image(current_image_path, points_path_global, labels_path_global)
        cv2.imshow("Lane Viewer", updated)

def update_label_for_selected_lane(new_label):
    global labels
    if 0 <= selected_lane_index < len(labels):
        labels[selected_lane_index] = new_label
        # Ghi lại file labels
        with open(labels_path_global, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        print(f"Đã cập nhật lane {selected_lane_index} thành nhãn {new_label}")
        redraw_image()
    else:
        print("Chưa chọn lane để sửa nhãn.")

def main(folder_path):
    global current_image_path
    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError("Không tìm thấy ảnh trong thư mục.")

    index = 0
    while True:
        current_image_path = image_paths[index]
        basename = os.path.splitext(os.path.basename(current_image_path))[0]
        points_path = os.path.join(folder_path, f"{basename}.lines.txt")
        labels_path = os.path.join(folder_path, f"{basename}.labels.txt")

        try:
            image = draw_lanes_on_image(current_image_path, points_path, labels_path)
        except Exception as e:
            print(f"[Lỗi] {current_image_path}: {e}")
            image = cv2.imread(current_image_path)

        title = f"{index+1}/{len(image_paths)}: {basename}.jpg"
        cv2.imshow("Lane Viewer", image)
        cv2.setWindowTitle("Lane Viewer", title)
        cv2.setMouseCallback("Lane Viewer", mouse_callback)

        key = cv2.waitKeyEx(0)

        if key == 27:  # ESC
            break
        elif key == 2424832:  # ←
            index = (index - 1) % len(image_paths)
            selected_lane_index = -1
        elif key == 2555904:  # →
            index = (index + 1) % len(image_paths)
            selected_lane_index = -1
        elif key == ord('0'):
            update_label_for_selected_lane(0)
        elif key == ord('1'):
            update_label_for_selected_lane(1)
        else:
            print(f"Nhấn phím: {key}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("D:\\lane_labeling_tool\\gt_image")
