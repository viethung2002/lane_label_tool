import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import random
import os
import json  # Import json module

# Ensure required directories exist
os.makedirs('gt_image', exist_ok=True)
os.makedirs('gt_binary_image', exist_ok=True)
os.makedirs('gt_instance_image', exist_ok=True)
os.makedirs('annotations', exist_ok=True)  # Ensure annotations directory exists


class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ chú thích ảnh")
        self.root.geometry("1000x700")

        # Main variables for the tool
        self.drawing = False
        self.mode = 'line'  # Khởi tạo chế độ 'line'
        self.action_mode = 'label'  # 'label', 'pan', 'edit'
        self.edit_sub_mode = None  # 'delete', 'add'
        self.ix, self.iy = -1, -1
        self.annotations = []  # Now a list of dictionaries
        self.current_object = []
        self.current_color = (0, 0, 255)
        self.thickness = 2
        self.img = None
        self.binary_img = None
        self.gray_lane_img = None
        self.image_path_list = []
        self.index = 0
        self.img_path = None if not self.image_path_list else self.image_path_list[self.index]
        self.text = f"0/0"
        self.check_save = [False]*len(self.image_path_list)
        self.image_label = {}
        self.selected_object_index = None
        self.object_counter = 1  # Initialize object counter
        self.image_annotations = {}  # To store annotations per image
        self.image_label = {}        # To store annotated images per image
        self.current_point = None  # Initialize current_point

        # Variables for zoom and pan
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Setup UI
        self.setup_ui()

        # Bind keyboard shortcuts
        self.bind_shortcuts()

    def setup_ui(self):
        # Toolbar
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        tk.Button(toolbar, text="Mở ảnh", command=self.open_folder_and_get_image_paths).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Lưu ảnh", command=self.save_images).pack(side=tk.LEFT, padx=2, pady=2)
        self.mode_button = tk.Button(toolbar, text="Chế độ vẽ: line", command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Hoàn thành đối tượng", command=self.finish_current_object).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Xóa gần nhất", command=self.undo_last_annotation).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Xóa tất cả", command=self.clear_all_annotations).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Tạo Train/Valid/Test", command=self.create_train_valid_test_files).pack(side=tk.LEFT, padx=2, pady=2)
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)
        # Label để hiển thị số lượng ảnh
        self.image_count_label = tk.Label(self.root, text=self.text, font=("Arial", 14))
        self.image_count_label.pack(side=tk.BOTTOM)

        # Checkbox để kiểm tra đã lưu
        self.save_check_var = tk.BooleanVar()
        self.save_check_box = tk.Checkbutton(self.root, text="Saved", variable=self.save_check_var, command=self.check_click_button)
        self.save_check_box.pack(side=tk.BOTTOM, anchor=tk.SE)

        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Thickness slider
        thickness_slider = tk.Scale(self.root, from_=1, to=30, orient=tk.HORIZONTAL, label="Độ dày nét vẽ",
                                    command=self.update_thickness)
        thickness_slider.set(self.thickness)
        thickness_slider.pack(side=tk.TOP, fill=tk.X)

        # Canvas for image display
        self.canvas = tk.Canvas(self.root, width=1640, height=590)
        self.canvas.pack()
        self.img_display = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Mouse event bindings for the canvas
        self.canvas.bind("<ButtonPress-1>", self.on_left_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_left_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self.on_right_mouse_down)
        self.canvas.bind("<B3-Motion>", self.on_right_mouse_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_mouse_up)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Motion>", self.on_mouse_motion)

        self.action_mode_button = tk.Button(toolbar, text="Chế độ: Label", command=self.toggle_action_mode)
        self.action_mode_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Nút chuyển đổi chế độ chỉnh sửa
        self.edit_mode_button = tk.Button(toolbar, text="Chỉnh sửa: None", command=self.toggle_edit_sub_mode, state=tk.DISABLED)
        self.edit_mode_button.pack(side=tk.LEFT, padx=2, pady=2)

    def bind_shortcuts(self):
        # Bind keyboard shortcuts to functions
        self.root.bind('<Control-o>', self.open_folder_and_get_image_paths)
        self.root.bind('<Control-s>', self.save_images)
        self.root.bind('m', self.toggle_mode)
        self.root.bind('<Tab>', self.toggle_action_mode)
        self.root.bind('<Return>', self.finish_current_object)
        self.root.bind('z', self.undo_last_annotation)
        self.root.bind('c', self.clear_all_annotations)
        self.root.bind('t', self.create_train_valid_test_files)
        self.root.bind('n', self.next_image)
        self.root.bind('p', self.prev_image)
        self.root.bind('d', self.set_edit_sub_mode_delete)
        self.root.bind('a', self.set_edit_sub_mode_add)

    def set_edit_sub_mode_delete(self, event=None):
        if self.action_mode == 'edit':
            self.edit_sub_mode = 'delete'
            self.edit_mode_button.config(text="Chỉnh sửa: Xóa đối tượng")
            self.selected_object_index = None
            self.update_display()

    def set_edit_sub_mode_add(self, event=None):
        if self.action_mode == 'edit':
            self.edit_sub_mode = 'add'
            self.edit_mode_button.config(text="Chỉnh sửa: Thêm vào đối tượng")
            self.selected_object_index = None
            self.update_display()

    def toggle_edit_sub_mode(self, event=None):
        if self.action_mode == 'edit':
            if self.edit_sub_mode == 'delete':
                self.edit_sub_mode = 'add'
            else:
                self.edit_sub_mode = 'delete'
            self.edit_mode_button.config(text=f"Chỉnh sửa: {'Xóa đối tượng' if self.edit_sub_mode == 'delete' else 'Thêm vào đối tượng'}")
            self.selected_object_index = None
            self.update_display()

    def create_instance_image(self, binary_image):
        # Tạo một instance image với cùng kích thước như binary_image
        instance_image = np.zeros_like(binary_image)

        # Tìm các thành phần liên thông
        num_labels, labels_im = cv2.connectedComponents(binary_image.astype(np.uint8))

        # Gán nhãn cho từng thành phần
        for label in range(1, num_labels):
            instance_image[labels_im == label] = label  # Gán nhãn cho các pixel tương ứng

        return instance_image

    def open_folder_and_get_image_paths(self, event=None):
        folder_path = filedialog.askdirectory()

        if folder_path:
            valid_extensions = ('.png', '.jpg', '.jpeg')
            self.image_path_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                                    if file.lower().endswith(valid_extensions)]
            self.index = 0  # Reset image index

        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

        self.img_path = self.image_path_list[self.index]
        self.check_save = [False]*len(self.image_path_list)

        if self.img_path:
            self.load_image_and_annotations()

    def load_image_and_annotations(self):
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        annotations_path = f"annotations/{base_name}.json"

        self.img = cv2.imread(self.img_path)
        if self.img is not None:
            self.img = cv2.resize(self.img, (1640, 590))
            self.binary_img = np.zeros((590, 1640), dtype=np.uint8)
            self.gray_lane_img = np.zeros((590, 1640), dtype=np.uint8)
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r') as f:
                    self.annotations = json.load(f)
                self.redraw_annotations()
            else:
                self.annotations = []
            self.update_display()
        else:
            messagebox.showerror("Error", "Cannot open image. Please try again.")

    def check_click_button(self):
        if not self.save_check_var.get():
            self.check_save[self.index] = False
        else:
            self.check_save[self.index] = True

    def next_image(self, event=None):
        if self.img_path is not None:
            self.image_annotations[self.img_path] = self.annotations.copy()
            self.image_label[self.img_path] = self.img.copy()
        if self.index < len(self.image_path_list) - 1:
            self.index += 1
            self.img_path = self.image_path_list[self.index]
            if not self.check_save[self.index]:
                self.save_check_var.set(False)
            else:
                self.save_check_var.set(True)
            self.load_image_and_annotations()
            self.zoom_scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
        else:
            messagebox.showerror("Error", "Cannot open image. Please try again.")
        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

    def prev_image(self, event=None):
        if self.img_path is not None:
            self.image_annotations[self.img_path] = self.annotations.copy()
            self.image_label[self.img_path] = self.img.copy()
        if self.index > 0:
            self.index -= 1
            self.img_path = self.image_path_list[self.index]
            if not self.check_save[self.index]:
                self.save_check_var.set(False)
            else:
                self.save_check_var.set(True)
            self.load_image_and_annotations()
            self.zoom_scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
        else:
            messagebox.showerror("Error", "Cannot open image. Please try again.")
        self.text = f"{self.index+1} / {len(self.image_path_list)}"
        self.image_count_label.config(text=self.text)

    def generate_points_from_line(self, line):
        x1, y1 = line[0]
        x2, y2 = line[1]
        points = []

        # Calculate the slope and intercept of the line
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        intercept = y1 - slope * x1

        # Ensure y values are rounded to the nearest multiple of 10
        y_start = round(y1 / 10) * 10
        y_end = round(y2 / 10) * 10

        # Determine the range for the y-axis based on the direction of drawing
        if y_start < y_end:
            y_range = np.arange(y_start, y_end + 10, 10)
        else:
            y_range = np.arange(y_start, y_end - 10, -10)  # Reverse the range if drawing from bottom to top

        for y in y_range:
            if slope != float('inf'):
                x = (y - intercept) / slope
            else:
                x = x1
            # Keep x as float and round y to the nearest multiple of 10
            points.append((x, y))
        
        return points


    def generate_and_save_points(self, base_name):
        points_per_line = []

        for obj in self.annotations:
            for annotation in obj['annotations']:
                if annotation[0] == 'line':
                    line = [annotation[1], annotation[2]]  # Get the start and end coordinates
                    points = self.generate_points_from_line(line)
                    points_per_line.append(points)

        # Write points to a text file
        txt_path = f"gt_image/{base_name}_lines.txt"
        with open(txt_path, "w") as file:
            for line_points in points_per_line:
                # Sort points by y in descending order
                line_points_sorted = sorted(line_points, key=lambda point: point[1], reverse=True)
                # Convert the list of points into a single string, where each point is separated by a space
                line_str = " ".join([f"{point[0]:.4f} {point[1]}" for point in line_points_sorted])
                file.write(line_str + "\n")  # Write each lane's points as a single line

    def save_images(self, event=None):
        while True:
            base_name = os.path.splitext(os.path.basename(self.image_path_list[self.index]))[0]

            # Define paths to save images
            image_path = f"gt_image/{base_name}.png"
            binary_path = f"gt_binary_image/{base_name}.png"
            gray_lane_path = f"gt_instance_image/{base_name}.png"
            annotations_path = f"annotations/{base_name}.json"  # Path for annotations file

            # Check if any file already exists
            if os.path.exists(image_path) or os.path.exists(binary_path) or os.path.exists(gray_lane_path) or os.path.exists(annotations_path):
                retry = messagebox.askyesno("File Exists", f"File '{base_name}' already exists. Overwrite?")
                if not retry:
                    return
            self.image_label[self.img_path] = self.img.copy()
            # Save the annotated original image
            img = cv2.imread(self.img_path)
            img = cv2.resize(img, (1640, 590))
            if self.img is not None:
                cv2.imwrite(image_path, img)

            # Save the binary image
            if self.binary_img is not None:
                cv2.imwrite(binary_path, self.binary_img)

            # Save the instance (gray lane) image
            if self.gray_lane_img is not None:
                cv2.imwrite(gray_lane_path, self.gray_lane_img)

            # Save annotations to JSON file
            with open(annotations_path, 'w') as f:
                json.dump(self.annotations, f)

            # Generate and save points to a text file
            self.generate_and_save_points(base_name)

            messagebox.showinfo("Save Image", f"Saved image as:\n{image_path}\n{binary_path}\n{gray_lane_path}\n{annotations_path}")
            self.check_save[self.index] = True
            self.save_check_var.set(True)
            break

    def create_train_valid_test_files(self, event=None):
        # Ask for base path
        base_path = simpledialog.askstring("Nhập đường dẫn",
                                           "Nhập đường dẫn cơ sở cho các tệp (ví dụ: dataset/dataset_tusimple/training):")
        if not base_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đường dẫn cơ sở.")
            return

        # Get n_sample for the test set
        n_sample = simpledialog.askinteger("Nhập n_sample", "Số lượng ảnh trong tập val:")
        if not n_sample:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập số lượng n_sample.")
            return

        # Get list of image base names from 'gt_image' folder
        img_files = sorted([f.split('.')[0] for f in os.listdir('gt_image') if f.endswith('.png')])
        print(img_files)

        if len(img_files) < n_sample:
            messagebox.showerror("Lỗi", "Số lượng ảnh trong thư mục ít hơn số lượng yêu cầu cho test.")
            return

        # Split data
        train_files = img_files

        # Write to files
        self.write_to_txt('train.txt', train_files, base_path)

        # Random selection for test set
        test_files = random.sample(img_files, n_sample)
        self.write_to_txt('val.txt', test_files, base_path)

        messagebox.showinfo("Hoàn tất", "Các tệp train.txt, val.txt đã được tạo.")

    def write_to_txt(self, filename, files_list, base_path):
        with open(filename, 'w') as f:
            for file_base in files_list:
                image_path = f"{base_path}/gt_image/{file_base}.png"
                binary_path = f"{base_path}/gt_binary_image/{file_base}.png"
                instance_path = f"{base_path}/gt_instance_image/{file_base}.png"
                f.write(f"{image_path} {binary_path} {instance_path}\n")

    def random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def random_gray(self):
        gray_value = random.randint(50, 200)  # Chọn một giá trị xám ngẫu nhiên (giữa 50 và 200 để dễ phân biệt)
        return gray_value

    def open_image(self, event=None):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.img_path = file_path
            self.img = cv2.imread(file_path)
            if self.img is not None:
                self.img = cv2.resize(self.img, (1640, 590))
                # Khởi tạo canvas nhị phân và canvas phân biệt lane cùng kích thước
                self.binary_img = np.zeros((590, 1640), dtype=np.uint8)
                self.gray_lane_img = np.zeros((590, 1640), dtype=np.uint8)
                self.update_display()
            else:
                messagebox.showerror("Lỗi", "Không thể mở ảnh. Vui lòng thử lại.")

    def update_display(self):
        if self.img is not None:
            # Thay đổi kích thước ảnh theo zoom_scale
            new_width = int(1640 * self.zoom_scale)
            new_height = int(590 * self.zoom_scale)
            img_resized = cv2.resize(self.img, (new_width, new_height))
            # Tạo bản sao để vẽ các chú thích
            img_display = img_resized.copy()
            # Vẽ các đối tượng
            for idx, obj in enumerate(self.annotations):
                # Nếu đối tượng đang được chọn, vẽ với màu đặc biệt
                if idx == self.selected_object_index:
                    color = (0, 255, 255)  # Màu vàng
                else:
                    color = None
                self.draw_object_on_image(obj, img_display, color)
            # Chuyển đổi ảnh từ BGR sang RGB
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            # Tạo ảnh PIL từ mảng NumPy
            img_pil = Image.fromarray(img_rgb)
            # Tạo đối tượng PhotoImage từ ảnh PIL
            img_tk = ImageTk.PhotoImage(image=img_pil)
            # Cập nhật hình ảnh trên canvas
            self.canvas.itemconfig(self.img_display, image=img_tk)
            self.canvas.image = img_tk
            # Di chuyển ảnh theo pan_x và pan_y
            self.canvas.coords(self.img_display, self.pan_x, self.pan_y)
        else:
            messagebox.showwarning("Warning", "No image to display!")  # Thông báo nếu không có ảnh

    def update_display_with_cursor(self, x, y):
        if self.img is None:
            return
        # Chuyển đổi tọa độ canvas sang tọa độ ảnh
        img_x = (x - self.pan_x) / self.zoom_scale
        img_y = (y - self.pan_y) / self.zoom_scale
        img_x = int(img_x)
        img_y = int(img_y)
        if self.img_path in self.image_label:
            img_temp = self.image_label[self.img_path].copy()
        else:
            img_temp = self.img.copy()
        cv2.circle(img_temp, (img_x, img_y), self.thickness // 2, self.current_color, -1)
        self.update_display_image(img_temp)

    def on_mouse_motion(self, event):
        if self.action_mode == 'label' and not self.drawing:
            self.update_display_with_cursor(event.x, event.y)
        elif self.action_mode == 'edit' and self.selected_object_index is not None and self.edit_sub_mode == 'add':
            self.update_display()
        else:
            pass

    def on_left_mouse_down(self, event):
        if self.action_mode == 'edit':
            self.select_object(event)
            if self.edit_sub_mode == 'delete' and self.selected_object_index is not None:
                self.delete_selected_object()
            elif self.edit_sub_mode == 'add' and self.selected_object_index is not None:
                x = int((event.x - self.pan_x) / self.zoom_scale)
                y = int((event.y - self.pan_y) / self.zoom_scale)
                self.drawing = True
                if self.mode == 'line':
                    self.current_object = [(x, y)]
                elif self.mode == 'curve':
                    self.current_object = [(x, y)]
                elif self.mode in ['polyline', 'polygon']:
                    if self.current_point is None:
                        self.current_point = (x, y)
                        self.current_object = []
                    else:
                        segment = {
                            'start': self.current_point,
                            'end': (x, y),
                            'color': self.current_color,
                            'thickness': self.thickness
                        }
                        self.current_object.append(segment)
                        self.current_point = (x, y)
                    self.redraw_current_object()
        else:
            x = (event.x - self.pan_x) / self.zoom_scale
            y = (event.y - self.pan_y) / self.zoom_scale
            x = int(x)
            y = int(y)
            self.drawing = True
            if self.mode == 'line':
                # Start drawing a line
                self.current_object = [(x, y)]
            elif self.mode == 'curve':
                # Start drawing a curve
                self.current_object = [(x, y)]
            elif self.mode in ['polyline', 'polygon']:
                if self.current_point is None:
                    # First point
                    self.current_point = (x, y)
                    self.current_object = []  # Initialize as empty list of segments
                else:
                    # Create a segment from current_point to new point
                    segment = {
                        'start': self.current_point,
                        'end': (x, y),
                        'color': self.current_color,
                        'thickness': self.thickness
                    }
                    self.current_object.append(segment)
                    self.current_point = (x, y)
                # Redraw current object
                self.redraw_current_object()

    def on_left_mouse_move(self, event):
        if self.drawing:
            x = int((event.x - self.pan_x) / self.zoom_scale)
            y = int((event.y - self.pan_y) / self.zoom_scale)
            if self.action_mode == 'edit' and self.edit_sub_mode == 'add':
                if self.mode == 'line':
                    img_temp = self.img.copy()
                    self.redraw_annotations(img=img_temp)
                    obj = self.annotations[self.selected_object_index]
                    cv2.line(img_temp, self.current_object[0], (x, y), self.current_color, self.thickness)
                    self.update_display_image(img_temp)
                elif self.mode == 'curve':
                    self.current_object.append((x, y))
                    img_temp = self.img.copy()
                    self.redraw_annotations(img=img_temp)
                    obj = self.annotations[self.selected_object_index]
                    for part in obj['annotations']:
                        if part[0] == 'curve':
                            pts = np.array(part[1] + self.current_object, np.int32).reshape((-1, 1, 2))
                        else:
                            pts = np.array(self.current_object, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_temp, [pts], isClosed=False, color=self.current_color, thickness=self.thickness)
                    self.update_display_image(img_temp)
                elif self.mode in ['polyline', 'polygon']:
                    if self.current_point:
                        img_temp = self.img.copy()
                        self.redraw_annotations(img=img_temp)
                        obj = self.annotations[self.selected_object_index]
                        for segment in self.current_object:
                            cv2.line(img_temp, segment['start'], segment['end'], segment['color'], segment['thickness'])
                        cv2.line(img_temp, self.current_point, (x, y), self.current_color, self.thickness)
                        self.update_display_image(img_temp)
            else:
                x = (event.x - self.pan_x) / self.zoom_scale
                y = (event.y - self.pan_y) / self.zoom_scale
                x = int(x)
                y = int(y)
                if self.mode == 'line':
                    # Vẽ đường thẳng tạm thời từ điểm bắt đầu đến vị trí chuột hiện tại
                    img_temp = self.img.copy()
                    cv2.line(img_temp, self.current_object[0], (x, y), self.current_color, self.thickness)
                    self.update_display_image(img_temp)
                elif self.mode == 'curve':
                    # Thêm điểm vào đường cong và vẽ tạm thời
                    self.current_object.append((x, y))
                    img_temp = self.img.copy()
                    pts = np.array(self.current_object, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_temp, [pts], isClosed=False, color=self.current_color, thickness=self.thickness)
                    self.update_display_image(img_temp)
                elif self.mode in ['polyline', 'polygon']:
                    if self.current_point:
                        img_temp = self.img.copy()
                        # Draw existing segments
                        for segment in self.current_object:
                            cv2.line(img_temp, segment['start'], segment['end'], segment['color'], segment['thickness'])
                        # Draw temporary line from current_point to (x, y)
                        cv2.line(img_temp, self.current_point, (x, y), self.current_color, self.thickness)
                        self.update_display_image(img_temp)

    def on_left_mouse_up(self, event):
        if self.drawing:
            x = int((event.x - self.pan_x) / self.zoom_scale)
            y = int((event.y - self.pan_y) / self.zoom_scale)
            if self.action_mode == 'edit' and self.edit_sub_mode == 'add':
                if self.mode == 'line':
                    self.current_object.append((x, y))
                    self.add_annotation_to_selected_object()
                    self.drawing = False
                elif self.mode == 'curve':
                    self.add_annotation_to_selected_object()
                    self.drawing = False
                elif self.mode in ['polyline', 'polygon']:
                    # Wait until the user finishes the object
                    pass
            else:
                x = (event.x - self.pan_x) / self.zoom_scale
                y = (event.y - self.pan_y) / self.zoom_scale
                x = int(x)
                y = int(y)
                if self.mode == 'line':
                    # Kết thúc vẽ đường thẳng
                    self.current_object.append((x, y))
                    self.finish_current_object()
                elif self.mode == 'curve':
                    # Kết thúc vẽ đường cong
                    self.finish_current_object()
                self.drawing = False

    def add_annotation_to_selected_object(self):
        obj = self.annotations[self.selected_object_index]
        gray_value = obj['gray_value']
        if self.mode == 'line' and len(self.current_object) == 2:
            x1, y1 = self.current_object[0]
            x2, y2 = self.current_object[1]
            cv2.line(self.img, (x1, y1), (x2, y2), self.current_color, self.thickness)
            cv2.line(self.binary_img, (x1, y1), (x2, y2), 255, self.thickness)
            cv2.line(self.gray_lane_img, (x1, y1), (x2, y2), gray_value, self.thickness)
            obj['annotations'].append(('line', (x1, y1), (x2, y2), self.current_color, self.thickness))
            self.current_object = []
            self.current_color = self.random_color()
            self.update_display()
        elif self.mode == 'curve' and len(self.current_object) >= 2:
            pts = np.array(self.current_object, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.img, [pts], isClosed=False, color=self.current_color, thickness=self.thickness)
            cv2.polylines(self.binary_img, [pts], isClosed=False, color=255, thickness=self.thickness)
            cv2.polylines(self.gray_lane_img, [pts], isClosed=False, color=gray_value, thickness=self.thickness)
            obj['annotations'].append(('curve', self.current_object.copy(), self.current_color, self.thickness))
            self.current_object = []
            self.current_color = self.random_color()
            self.update_display()
        elif self.mode == 'polygon' and len(self.current_object) >= 3:
            pts = np.array(self.current_object, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.img, [pts], isClosed=True, color=self.current_color, thickness=self.thickness)
            cv2.fillPoly(self.img, [pts], color=self.current_color)
            cv2.fillPoly(self.binary_img, [pts], color=255)
            cv2.fillPoly(self.gray_lane_img, [pts], color=gray_value)
            obj['annotations'].append(('polygon', self.current_object.copy(), self.current_color, self.thickness))
            self.current_object = []
            self.current_color = self.random_color()
            self.update_display()
        elif self.mode == 'polyline' and len(self.current_object) >= 1:
            for segment in self.current_object:
                cv2.line(self.img, segment['start'], segment['end'], segment['color'], segment['thickness'])
                cv2.line(self.binary_img, segment['start'], segment['end'], 255, segment['thickness'])
                cv2.line(self.gray_lane_img, segment['start'], segment['end'], gray_value, segment['thickness'])
            obj['annotations'].append(('polyline', self.current_object.copy()))
            self.current_object = []
            self.current_point = None
            self.current_color = self.random_color()
            self.update_display()
        else:
            messagebox.showwarning("Cảnh báo", "Cần nhiều điểm hơn để tạo đối tượng.")

    def redraw_current_object(self):
        img_temp = self.img.copy()
        if self.current_object:
            for segment in self.current_object:
                cv2.line(img_temp, segment['start'], segment['end'], segment['color'], segment['thickness'])
        self.update_display_image(img_temp)

    def on_right_mouse_down(self, event):
        self.start_pan(event)

    def on_right_mouse_move(self, event):
        self.pan_image(event)

    def on_right_mouse_up(self, event):
        pass  # Không cần làm gì khi nhả chuột trong chế độ pan

    def zoom(self, event):
        # Lấy vị trí chuột trên canvas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        # Lấy tọa độ ảnh trước khi zoom
        image_x = (x - self.pan_x) / self.zoom_scale
        image_y = (y - self.pan_y) / self.zoom_scale
        # Điều chỉnh zoom_scale
        if event.delta > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale /= 1.1
        self.zoom_scale = max(min(self.zoom_scale, 10), 0.1)
        # Điều chỉnh pan để giữ điểm dưới con trỏ chuột cố định
        self.pan_x = x - image_x * self.zoom_scale
        self.pan_y = y - image_y * self.zoom_scale
        self.update_display()

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan_image(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.update_display()

    def draw_annotation(self, event):
        # Convert canvas coordinates to image coordinates
        x = (event.x - self.pan_x) / self.zoom_scale
        y = (event.y - self.pan_y) / self.zoom_scale
        x = int(x)
        y = int(y)

        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi vẽ.")
            return

        # Append the new point
        self.current_object.append((x, y))

        # Redraw the temporary object on a copy of the image
        img_temp = self.img.copy()
        if len(self.current_object) > 1:
            pts = np.array(self.current_object, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_temp, [pts], isClosed=False,
                          color=self.current_color, thickness=self.thickness)
        for pt in self.current_object:
            cv2.circle(img_temp, pt, self.thickness, self.current_color, -1)

        # Update the display with the temporary drawing
        self.update_display_image(img_temp)

    def finish_current_object(self, event=None):
        if self.action_mode == 'edit' and self.edit_sub_mode == 'add' and self.selected_object_index is not None:
            self.add_annotation_to_selected_object()
        else:
            if self.mode == 'line':
                if len(self.current_object) == 2:
                    x1, y1 = self.current_object[0]
                    x2, y2 = self.current_object[1]
                    cv2.line(self.img, (x1, y1), (x2, y2), self.current_color, self.thickness)
                    cv2.line(self.binary_img, (x1, y1), (x2, y2), 255, self.thickness)
                    gray_value = self.random_gray()
                    cv2.line(self.gray_lane_img, (x1, y1), (x2, y2), gray_value, self.thickness)
                    object_id = self.object_counter
                    self.object_counter += 1
                    obj = {
                        'id': object_id,
                        'annotations': [('line', (x1, y1), (x2, y2), self.current_color, self.thickness)],
                        'gray_value': gray_value
                    }
                    self.annotations.append(obj)
                    self.current_object = []
                    self.current_color = self.random_color()
                    self.update_display()
                else:
                    messagebox.showwarning("Cảnh báo", "Cần 2 điểm để tạo một đường thẳng.")
            elif self.mode == 'curve':
                if len(self.current_object) >= 2:
                    pts = np.array(self.current_object, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(self.img, [pts], isClosed=False, color=self.current_color, thickness=self.thickness)
                    cv2.polylines(self.binary_img, [pts], isClosed=False, color=255, thickness=self.thickness)
                    gray_value = self.random_gray()
                    cv2.polylines(self.gray_lane_img, [pts], isClosed=False, color=gray_value, thickness=self.thickness)
                    object_id = self.object_counter
                    self.object_counter += 1
                    obj = {
                        'id': object_id,
                        'annotations': [('curve', self.current_object.copy(), self.current_color, self.thickness)],
                        'gray_value': gray_value
                    }
                    self.annotations.append(obj)
                    self.current_object = []
                    self.current_color = self.random_color()
                    self.update_display()
                else:
                    messagebox.showwarning("Cảnh báo", "Cần ít nhất 2 điểm để tạo một đường cong.")
            elif self.mode == 'polygon':
                if len(self.current_object) >= 3:
                    pts = np.array(self.current_object, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(self.img, [pts], isClosed=True, color=self.current_color, thickness=self.thickness)
                    cv2.fillPoly(self.img, [pts], color=self.current_color)
                    cv2.fillPoly(self.binary_img, [pts], color=255)
                    gray_value = self.random_gray()
                    cv2.fillPoly(self.gray_lane_img, [pts], color=gray_value)
                    object_id = self.object_counter
                    self.object_counter += 1
                    obj = {
                        'id': object_id,
                        'annotations': [('polygon', self.current_object.copy(), self.current_color, self.thickness)],
                        'gray_value': gray_value
                    }
                    self.annotations.append(obj)
                    self.current_object = []
                    self.current_color = self.random_color()
                    self.redraw_annotations()
                    self.update_display()
                else:
                    messagebox.showwarning("Cảnh báo", "Cần ít nhất 3 điểm để tạo một vùng.")
            elif self.mode == 'polyline':
                if len(self.current_object) >= 1:
                    # Draw the segments onto the images
                    gray_value = self.random_gray()
                    for segment in self.current_object:
                        cv2.line(self.img, segment['start'], segment['end'], segment['color'], segment['thickness'])
                        cv2.line(self.binary_img, segment['start'], segment['end'], 255, segment['thickness'])
                        cv2.line(self.gray_lane_img, segment['start'], segment['end'], gray_value, segment['thickness'])
                    object_id = self.object_counter
                    self.object_counter += 1
                    obj = {
                        'id': object_id,
                        'annotations': [('polyline', self.current_object.copy())],
                        'gray_value': gray_value
                    }
                    self.annotations.append(obj)
                    self.current_object = []
                    self.current_point = None
                    self.current_color = self.random_color()
                    self.update_display()
                else:
                    messagebox.showwarning("Cảnh báo", "Cần ít nhất 2 điểm để tạo một đường gấp khúc.")

    def toggle_mode(self, event=None):
        modes = ['line', 'curve', 'polygon', 'polyline']
        current_index = modes.index(self.mode)
        self.mode = modes[(current_index + 1) % len(modes)]
        self.mode_button.config(text=f"Chế độ vẽ: {self.mode}")

    def toggle_action_mode(self, event=None):
        if self.action_mode == 'label':
            self.action_mode = 'pan'
            self.action_mode_button.config(text="Chế độ: Pan")
            self.edit_mode_button.config(state=tk.DISABLED)
            self.edit_sub_mode = None
            self.selected_object_index = None
            self.update_display()
        elif self.action_mode == 'pan':
            self.action_mode = 'edit'
            self.action_mode_button.config(text="Chế độ: Chỉnh sửa")
            self.edit_mode_button.config(state=tk.NORMAL)
            self.edit_sub_mode = 'delete'  # Mặc định là 'delete'
            self.edit_mode_button.config(text="Chỉnh sửa: Xóa đối tượng")
        else:
            self.action_mode = 'label'
            self.action_mode_button.config(text="Chế độ: Label")
            self.edit_mode_button.config(state=tk.DISABLED)
            self.edit_sub_mode = None
            self.selected_object_index = None
            self.update_display()

    def update_display_image(self, temp_img):
        # Resize the temporary image according to the zoom scale
        new_width = int(1640 * self.zoom_scale)
        new_height = int(590 * self.zoom_scale)
        img_resized = cv2.resize(temp_img, (new_width, new_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.itemconfig(self.img_display, image=img_tk)
        self.canvas.image = img_tk
        # Move the image according to pan_x and pan_y
        self.canvas.coords(self.img_display, self.pan_x, self.pan_y)

    def undo_last_annotation(self, event=None):
        if self.current_object:
            if self.mode in ['polyline', 'polygon']:
                if self.current_object:
                    self.current_object.pop()
                    if not self.current_object and self.current_point:
                        self.current_point = None
                    self.redraw_current_object()
            else:
                self.current_object.pop()
                self.redraw_current_object()
        elif self.annotations:
            if self.action_mode == 'edit' and self.edit_sub_mode == 'add' and self.selected_object_index is not None:
                obj = self.annotations[self.selected_object_index]
                if obj['annotations']:
                    last_annotation = obj['annotations'].pop()
                    self.redraw_annotations()
                    self.update_display()
                else:
                    messagebox.showinfo("Thông báo", "Không còn chú thích để xóa.")
            else:
                self.annotations.pop()
                self.redraw_annotations()
        else:
            messagebox.showinfo("Thông báo", "Không còn chú thích để xóa.")

    def clear_all_annotations(self, event=None):
        if self.img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở một ảnh trước khi xóa.")
            return

        if messagebox.askyesno("Xác nhận", "Bạn có chắc chắn muốn xóa tất cả các chú thích không?"):
            self.annotations = []
            self.current_object = []
            self.save_check_var.set(False)
            if self.img_path in self.image_label:
                del self.image_label[self.img_path]
            if self.img_path in self.image_annotations:
                del self.image_annotations[self.img_path]
            self.binary_img = np.zeros((590, 1640), dtype=np.uint8)  # Clear binary canvas
            self.gray_lane_img = np.zeros((590, 1640), dtype=np.uint8)  # Clear instance canvas
            self.redraw_annotations()

    def redraw_annotations(self, img=None):
        if self.img_path:
            if img is None:
                self.img = cv2.imread(self.img_path)
                if self.img is not None:
                    self.img = cv2.resize(self.img, (1640, 590))
                    self.binary_img = np.zeros((590, 1640), dtype=np.uint8)
                    self.gray_lane_img = np.zeros((590, 1640), dtype=np.uint8)
                else:
                    messagebox.showerror("Error", "Cannot reload original image.")
                    return
            else:
                self.img = img

            for obj in self.annotations:
                gray_value = obj['gray_value']
                for part in obj['annotations']:
                    if part[0] == 'line':
                        _, (x1, y1), (x2, y2), color, thickness = part
                        color = tuple(color)  # Convert color to tuple
                        cv2.line(self.img, (x1, y1), (x2, y2), color, thickness)
                        cv2.line(self.binary_img, (x1, y1), (x2, y2), 255, thickness)
                        cv2.line(self.gray_lane_img, (x1, y1), (x2, y2), gray_value, thickness)
                    elif part[0] == 'curve':
                        _, points, color, thickness = part
                        color = tuple(color)
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(self.img, [pts], isClosed=False, color=color, thickness=thickness)
                        cv2.polylines(self.binary_img, [pts], isClosed=False, color=255, thickness=thickness)
                        cv2.polylines(self.gray_lane_img, [pts], isClosed=False, color=gray_value, thickness=thickness)
                    elif part[0] == 'polygon':
                        _, points, color, thickness = part
                        color = tuple(color)
                        pts = np.array(points, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(self.img, [pts], isClosed=True, color=color, thickness=thickness)
                        cv2.fillPoly(self.img, [pts], color=color)
                        cv2.fillPoly(self.binary_img, [pts], color=255)
                        cv2.fillPoly(self.gray_lane_img, [pts], color=gray_value)
                    elif part[0] == 'polyline':
                        _, segments = part
                        for segment in segments:
                            start = segment['start']
                            end = segment['end']
                            color = tuple(segment['color'])
                            thickness = segment['thickness']
                            cv2.line(self.img, start, end, color, thickness)
                            cv2.line(self.binary_img, start, end, 255, thickness)
                            cv2.line(self.gray_lane_img, start, end, gray_value, thickness)
            self.update_display()

    def update_thickness(self, value):
        self.thickness = int(value)

    def select_object(self, event):
        x = int((event.x - self.pan_x) / self.zoom_scale)
        y = int((event.y - self.pan_y) / self.zoom_scale)
        for idx, obj in enumerate(self.annotations):
            if self.is_point_near_object((x, y), obj['annotations']):
                self.selected_object_index = idx
                self.update_display()
                return
        self.selected_object_index = None
        self.update_display()

    def is_point_near_object(self, point, annotations, threshold=5):
        x, y = point
        for part in annotations:
            if part[0] == 'polygon':
                points = part[1]
                contour = np.array(points, dtype=np.int32)
                dist = cv2.pointPolygonTest(contour, (x, y), False)
                if dist >= 0:
                    return True
            elif part[0] == 'line':
                points = [part[1], part[2]]
                for i in range(len(points) - 1):
                    start = points[i]
                    end = points[i + 1]
                    if self.is_point_near_line_segment((x, y), start, end, threshold):
                        return True
            elif part[0] == 'curve':
                points = part[1]
                for i in range(len(points) - 1):
                    start = points[i]
                    end = points[i + 1]
                    if self.is_point_near_line_segment((x, y), start, end, threshold):
                        return True
            elif part[0] == 'polyline':
                segments = part[1]
                for segment in segments:
                    start = segment['start']
                    end = segment['end']
                    if self.is_point_near_line_segment((x, y), start, end, threshold):
                        return True
        return False

    def is_point_near_line_segment(self, point, start, end, threshold):
        x, y = point
        x1, y1 = start
        x2, y2 = end
        # Compute the squared length of the line segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            # The segment is a point
            distance = np.hypot(x - x1, y - y1)
            return distance <= threshold
        else:
            # Project point onto the line segment
            t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            distance = np.hypot(x - proj_x, y - proj_y)
            return distance <= threshold

    def delete_selected_object(self, event=None):
        if self.selected_object_index is not None:
            del self.annotations[self.selected_object_index]
            self.selected_object_index = None
            self.redraw_annotations()
            self.update_display()

    def draw_object_on_image(self, obj, img, override_color=None):
        cX, cY = None, None  # Initialize centroid variables
        for part in obj['annotations']:
            if part[0] == 'line':
                _, (x1, y1), (x2, y2), color, thickness = part
                if override_color:
                    color = override_color
                x1 = int(x1 * self.zoom_scale)
                y1 = int(y1 * self.zoom_scale)
                x2 = int(x2 * self.zoom_scale)
                y2 = int(y2 * self.zoom_scale)
                cv2.line(img, (x1, y1), (x2, y2), color, int(thickness * self.zoom_scale))
                # Trọng tâm là trung điểm của đường thẳng
                cX = (x1 + x2) // 2
                cY = (y1 + y2) // 2
            elif part[0] == 'curve':
                _, points, color, thickness = part
                if override_color:
                    color = override_color
                scaled_points = [(int(x * self.zoom_scale), int(y * self.zoom_scale)) for x, y in points]
                cv2.polylines(img, [np.array(scaled_points)], isClosed=False, color=color, thickness=int(thickness * self.zoom_scale))
                # Tính trọng tâm của đường cong
                M = cv2.moments(np.array(scaled_points, dtype=np.int32))
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                else:
                    cX, cY = scaled_points[0]
            elif part[0] == 'polygon':
                _, points, color, thickness = part
                if override_color:
                    color = override_color
                scaled_points = [(int(x * self.zoom_scale), int(y * self.zoom_scale)) for x, y in points]
                cv2.polylines(img, [np.array(scaled_points)], isClosed=True, color=color, thickness=int(thickness * self.zoom_scale))
                # Tính trọng tâm của đa giác
                M = cv2.moments(np.array(scaled_points, dtype=np.int32))
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                else:
                    cX, cY = scaled_points[0]
            elif part[0] == 'polyline':
                _, segments = part
                for segment in segments:
                    start = segment['start']
                    end = segment['end']
                    color = segment['color']
                    thickness = segment['thickness']
                    if override_color:
                        color = override_color
                    x1 = int(start[0] * self.zoom_scale)
                    y1 = int(start[1] * self.zoom_scale)
                    x2 = int(end[0] * self.zoom_scale)
                    y2 = int(end[1] * self.zoom_scale)
                    cv2.line(img, (x1, y1), (x2, y2), color, int(thickness * self.zoom_scale))
                    # Compute centroid if needed
        if cX is not None and cY is not None:
            # Chuẩn bị văn bản ID
            id_text = f"ID: {obj['id']}"
            # Thiết lập các tham số font chữ
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * self.zoom_scale
            font_thickness = 1
            text_color = (255, 255, 255)  # Màu trắng
            text_bg_color = (0, 0, 0)     # Màu đen

            # Lấy kích thước văn bản
            (text_width, text_height), _ = cv2.getTextSize(id_text, font, font_scale, font_thickness)
            # Tính toán vị trí văn bản
            text_offset_x = cX - text_width // 2
            text_offset_y = cY - text_height // 2

            # Vẽ hình chữ nhật nền cho văn bản
            cv2.rectangle(img, (text_offset_x, text_offset_y - text_height),
                          (text_offset_x + text_width, text_offset_y + 5),
                          text_bg_color, cv2.FILLED)
            # Đặt văn bản ID lên ảnh
            cv2.putText(img, id_text, (text_offset_x, text_offset_y),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)


# Khởi chạy ứng dụng
root = tk.Tk()
app = AnnotationTool(root)
root.mainloop()
