import cv2
import numpy as np
import os
from pathlib import Path
import keyboard

def copy_label_files(input_label_file, output_dir, start_index, num_files):
    """
    Sao chép file nhãn gốc sang các file mới với số thứ tự tăng dần.
    - input_label_file: Đường dẫn đến file nhãn gốc (ví dụ: '0117.labels.txt')
    - output_dir: Thư mục để lưu các file nhãn mới
    - start_index: Số thứ tự bắt đầu (XXXX, ví dụ: 117 cho '0117')
    - num_files: Số lượng file nhãn cần tạo
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(input_label_file, 'r') as f:
            label_content = f.read()
        print(f"Read label file {input_label_file} with content:\n{label_content}")
    except Exception as e:
        print(f"Error reading {input_label_file}: {e}")
        return
    
    for i in range(start_index, start_index + num_files):
        output_file = os.path.join(output_dir, f"{i:04d}.labels.txt")
        try:
            with open(output_file, 'w') as f:
                f.write(label_content)
            print(f"Created label file: {output_file}")
        except Exception as e:
            print(f"Error creating {output_file}: {e}")

def visualize_lanes_directory(input_dir, output_dir):
    """
    Vẽ các làn đường từ file .lines.txt lên các ảnh trong thư mục với màu sắc dựa trên nhãn từ file .labels.txt.
    - Nhãn 0 (solid): Màu đỏ (0, 0, 255).
    - Nhãn 1 (dashed): Màu xanh (0, 255, 0).
    - Không có nhãn: Màu tím (128, 0, 128).
    - Điều hướng: Mũi tên trái/phải để chuyển ảnh, Ctrl+C hoặc Esc để thoát.
    
    Parameters:
    - input_dir: Thư mục chứa các file ảnh (*.jpg hoặc *.png) và file .lines.txt, .labels.txt
    - output_dir: Thư mục để lưu ảnh kết quả
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách file ảnh (*.jpg hoặc *.png)
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} image files: {image_files}")
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    current_idx = 0
    window_name = 'Visualized Lanes'
    
    try:
        while 0 <= current_idx < len(image_files):
            img_file = image_files[current_idx]
            image_path = os.path.join(input_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(input_dir, f"{base_name}.lines.txt")
            label_path = os.path.join(input_dir, f"{base_name}.labels.txt")
            
            # Kiểm tra sự tồn tại của file tọa độ
            if not os.path.exists(txt_path):
                print(f"Text file not found for {img_file}: {txt_path}")
                current_idx += 1
                continue
            
            # Đọc file tọa độ
            try:
                with open(txt_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
                current_idx += 1
                continue
            
            lanes = []
            for line in lines:
                coords = line.split()
                if len(coords) % 2 != 0 or len(coords) == 0:
                    print(f"Invalid coordinate format in {txt_path}, line: {line}")
                    continue
                lane_points = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
                lanes.append(lane_points)
            
            # Đọc file nhãn (nếu có)
            labels = None
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        labels = [int(line.strip()) for line in f if line.strip() and line.strip().isdigit()]
                    if len(labels) != len(lanes):
                        print(f"Number of labels ({len(labels)}) does not match number of lanes ({len(lanes)}) in {label_path}")
                        labels = None
                except Exception as e:
                    print(f"Error reading {label_path}: {e}")
                    labels = None
            else:
                print(f"Label file not found for {img_file}: {label_path}. Using purple color for lanes.")
            
            # Đọc ảnh gốc
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                current_idx += 1
                continue
            
            # Resize ảnh về kích thước chuẩn
            img = cv2.resize(img, (1640, 590))
            
            # Vẽ các làn đường
            for lane_idx, lane_points in enumerate(lanes):
                if labels is not None and lane_idx < len(labels):
                    color = (0, 255, 0) if labels[lane_idx] == 1 else (0, 0, 255)
                    label_text = f"Lane {lane_idx+1}: {'Dashed' if labels[lane_idx] == 1 else 'Solid'}"
                else:
                    color = (128, 0, 128)
                    label_text = f"Lane {lane_idx+1}: Unknown"
                
                points = np.array(lane_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [points], isClosed=False, color=color, thickness=2)
                
                if lane_points:
                    x, y = lane_points[0]
                    cv2.putText(img, label_text, (int(x), int(y)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            # Lưu ảnh kết quả
            output_path = os.path.join(output_dir, f"visualized_{img_file}")
            cv2.imwrite(output_path, img)
            print(f"Saved visualized image: {output_path}")
            
            # Hiển thị ảnh
            cv2.imshow(window_name, img)
            print(f"Displaying image {current_idx+1}/{len(image_files)}: {img_file}")
            print("Press 'Right Arrow' to go forward, 'Left Arrow' to go back, 'Esc' to exit")
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # Esc
                    raise KeyboardInterrupt
                elif key == ord('q'):  # Q key to exit
                    raise KeyboardInterrupt
                elif key == 2:  # Mũi tên trái
                    current_idx = max(0, current_idx - 1)
                    break
                elif key == 3:  # Mũi tên phải
                    current_idx = min(len(image_files) - 1, current_idx + 1)
                    break
            
    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        cv2.destroyAllWindows()

# Sử dụng
input_dir = r'D:\lane_labeling_tool\gt_image'
output_dir = r'D:\lane_labeling_tool\visualized_lanes'
input_label_file = r'D:\lane_labeling_tool\gt_image\0156.labels.txt'
start_index = 156
num_files = 20

# Bước 1: Sao chép file nhãn
print("Copying label files...")
copy_label_files(input_label_file, input_dir, start_index, num_files)

# Bước 2: Hiển thị và lưu các ảnh với làn đường
print("Visualizing lanes for all images...")
visualize_lanes_directory(input_dir, output_dir)
