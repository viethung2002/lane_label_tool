import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
from pathlib import Path

class LaneDataset(Dataset):
    def __init__(self, image_dir, txt_dir, label_dir, max_points=50, img_size=(1640, 590)):
        """
        Dataset để đọc ảnh, tọa độ làn đường, và nhãn.
        - image_dir: Thư mục chứa ảnh (gt_image).
        - txt_dir: Thư mục chứa file .txt (tọa độ làn đường).
        - label_dir: Thư mục chứa file _labels.txt (nhãn solid/dashed).
        - max_points: Số điểm tối đa cho mỗi làn đường.
        - img_size: Kích thước ảnh chuẩn.
        """
        self.image_dir = image_dir
        self.txt_dir = txt_dir
        self.label_dir = label_dir
        self.max_points = max_points
        self.img_size = img_size
        self.files = [f for f in os.listdir(txt_dir) if f.endswith('.lines.txt')]
        
        # Transform để chuẩn hóa ảnh
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        # Mỗi file .txt chứa nhiều làn đường
        total_lanes = 0
        for txt_file in self.files:
            with open(os.path.join(self.txt_dir, txt_file), 'r') as f:
                total_lanes += len(f.readlines())
        return total_lanes
    
    def __getitem__(self, idx):
        # Tìm file và lane tương ứng với idx
        current_idx = 0
        for txt_file in self.files:
            with open(os.path.join(self.txt_dir, txt_file), 'r') as f:
                lines = f.readlines()
            num_lanes = len(lines)
            if current_idx + num_lanes > idx:
                lane_idx = idx - current_idx
                break
            current_idx += num_lanes
        
        # Đọc tọa độ
        coords = lines[lane_idx].strip().split()
        lane_points = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
        
        # Chuẩn hóa tọa độ
        if len(lane_points) > self.max_points:
            indices = np.linspace(0, len(lane_points)-1, self.max_points).astype(int)
            lane_points = [lane_points[i] for i in indices]
        elif len(lane_points) < self.max_points:
            lane_points = lane_points + [(0, 0)] * (self.max_points - len(lane_points))
        coords = np.array(lane_points)
        coords[:, 0] = coords[:, 0] / self.img_size[0]  # Chuẩn hóa x
        coords[:, 1] = coords[:, 1] / self.img_size[1]  # Chuẩn hóa y
        coords = coords.flatten()
        
        # Đọc nhãn
        label_file = txt_file.replace('.lines.txt', '.labels.txt')
        label_path = os.path.join(self.label_dir, label_file)
        with open(label_path, 'r') as f:
            labels = [int(line.strip()) for line in f]
        label = labels[lane_idx]
        
        # Đọc và chuẩn hóa ảnh
        img_name = txt_file.replace('.lines.txt', '.jpg')
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        
        return img, torch.tensor(coords, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class LaneClassifier(nn.Module):
    def __init__(self, coord_dim=100, hidden_dim=128):
        """
        Mô hình phân loại làn đường.
        - coord_dim: Kích thước vector tọa độ (max_points * 2).
        - hidden_dim: Kích thước lớp ẩn.
        """
        super(LaneClassifier, self).__init__()
        # CNN để trích xuất đặc trưng từ ảnh
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Loại bỏ lớp fully connected cuối
        cnn_output_dim = 512  # ResNet18 output
        
        # MLP để xử lý tọa độ và kết hợp
        self.coord_fc = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Lớp kết hợp đặc trưng ảnh và tọa độ
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 lớp: solid (0), dashed (1)
        )
    
    def forward(self, img, coords):
        img_features = self.cnn(img)
        coord_features = self.coord_fc(coords)
        combined = torch.cat((img_features, coord_features), dim=1)
        output = self.fc(combined)
        return output

def train_model(dataset, batch_size=8, num_epochs=50, device='cuda'):
    """
    Huấn luyện mô hình.
    - dataset: LaneDataset.
    - batch_size: Kích thước batch.
    - num_epochs: Số epoch huấn luyện.
    - device: Thiết bị (cuda hoặc cpu).
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LaneClassifier(coord_dim=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for imgs, coords, labels in dataloader:
            imgs, coords, labels = imgs.to(device), coords.to(device), labels.to(device)
            outputs = model(imgs, coords)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100 * correct/total:.2f}%')
    
    # Lưu mô hình
    torch.save(model.state_dict(), os.path.join(dataset.image_dir, 'lane_classifier.pth'))
    return model

def predict(model, image_path, txt_path, label_path, device='cuda'):
    """
    Suy luận trên một ảnh cụ thể.
    - image_path: Đường dẫn đến ảnh.
    - txt_path: Đường dẫn đến file tọa độ.
    - label_path: Đường dẫn đến file nhãn (dùng để so sánh).
    """
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Đọc dữ liệu
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1640, 590))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    lanes = []
    for line in lines:
        coords = line.strip().split()
        lane_points = [(float(coords[i]), float(coords[i+1])) for i in range(0, len(coords), 2)]
        lanes.append(lane_points)
    
    with open(label_path, 'r') as f:
        true_labels = [int(line.strip()) for line in f]
    
    max_points = 50
    predictions = []
    for lane_points in lanes:
        if len(lane_points) > max_points:
            indices = np.linspace(0, len(lane_points)-1, max_points).astype(int)
            lane_points = [lane_points[i] for i in indices]
        elif len(lane_points) < max_points:
            lane_points = lane_points + [(0, 0)] * (max_points - len(lane_points))
        coords = np.array(lane_points)
        coords[:, 0] = coords[:, 0] / 1640
        coords[:, 1] = coords[:, 1] / 590
        coords = coords.flatten()
        coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img.to(device), coords)
            _, pred = torch.max(output, 1)
            predictions.append(pred.item())
    
    # In kết quả
    print(f"True labels: {true_labels}")
    print(f"Predicted labels: {predictions}")
    
    # Vẽ kết quả lên ảnh
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1640, 590))
    for lane_idx, (lane_points, pred) in enumerate(zip(lanes, predictions)):
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        points = np.array(lane_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=False, color=color, thickness=2)
        if lane_points:
            x, y = lane_points[0]
            cv2.putText(img, f"Lane {lane_idx+1}: {'Dashed' if pred == 1 else 'Solid'}", 
                        (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    output_path = os.path.join(os.path.dirname(image_path), 'visualized_lanes', f"predicted_{os.path.basename(image_path)}")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved predicted image: {output_path}")

# Sử dụng
image_dir = r'D:\lane_labeling_tool\gt_image'
txt_dir = r'D:\lane_labeling_tool\gt_image'
label_dir = r'D:\lane_labeling_tool\gt_image'
dataset = LaneDataset(image_dir, txt_dir, label_dir, max_points=50, img_size=(1640, 590))

# Huấn luyện mô hình
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = train_model(dataset, batch_size=8, num_epochs=50, device=device)

# Suy luận trên một ảnh cụ thể
image_path = r'D:\lane_labeling_tool\gt_image\0000.jpg'
txt_path = r'D:\lane_labeling_tool\gt_image\0000.lines.txt'
label_path = r'D:\lane_labeling_tool\gt_image\0000.labels.txt'
predict(model, image_path, txt_path, label_path, device=device)
