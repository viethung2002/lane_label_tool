import os
from pathlib import Path

def sort_lane_files(input_dir, output_dir):
    """
    Sắp xếp các dòng trong file XXXX.lines.txt theo tọa độ x đầu tiên.
    - input_dir: Thư mục chứa các file XXXX.lines.txt
    - output_dir: Thư mục để lưu các file đã sắp xếp
    """
    # Tạo thư mục đầu ra
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # In danh sách file trong thư mục
    all_files = os.listdir(input_dir)
    print(f"All files in {input_dir}: {all_files}")
    txt_files = sorted([f for f in all_files if f.endswith('.lines.txt')])
    print(f"Coordinate files found: {txt_files}")
    
    for txt_file in txt_files:
        txt_path = os.path.join(input_dir, txt_file)
        
        # Đọc file tọa độ
        try:
            with open(txt_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            continue
        
        # Tạo danh sách các dòng với tọa độ x đầu tiên
        lane_data = []
        for line in lines:
            coords = line.split()
            if len(coords) % 2 != 0 or len(coords) == 0:
                print(f"Invalid coordinate format in {txt_file}, line: {line}")
                continue
            x_first = float(coords[0])
            lane_data.append((x_first, line))
        
        # Sắp xếp theo x_first
        lane_data.sort(key=lambda x: x[0])
        
        # Lấy các dòng đã sắp xếp
        sorted_lines = [data[1] for data in lane_data]
        
        # Lưu file tọa độ đã sắp xếp
        sorted_txt_path = os.path.join(output_dir, txt_file)
        with open(sorted_txt_path, 'w') as f:
            f.write('\n'.join(sorted_lines) + '\n')
        print(f"Saved sorted coordinate file: {sorted_txt_path}")

# Sử dụng
input_dir = r'D:\lane_labeling_tool\gt_image'
output_dir = r'D:\lane_labeling_tool\gt_image\sorted'
sort_lane_files(input_dir, output_dir)
