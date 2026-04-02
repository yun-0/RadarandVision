# experiments/train_multi.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import AdvancedFusionModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 장치: {device}")

    # 1. 데이터 불러오기 (multi 데이터 폴더 확인!)
    current_dir = os.path.dirname(os.path.abspath(__file__)) # experiments 폴더
    project_root = os.path.join(current_dir, '..')           # 프로젝트 최상위 폴더
    data_dir = os.path.join(project_root, 'data', 'raw_crowded') # 정확한 데이터 경로

    full_dataset = RadarVisionDataset(data_dir, grid_size=4)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2. 모델 및 최적화 설정
    model = AdvancedFusionModel(mode='hybrid', grid_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Loss 함수 정의
    criterion_obj = nn.BCELoss() # 존재 여부
    criterion_box = nn.MSELoss() # 위치 정보

    epochs = 50
    best_val_loss = float('inf')
    save_path = os.path.join(project_root, 'experiments', 'model_multi_best.pth')

    print(f"🚀 [다중 객체 모드] 학습 시작! (Grid: 4x4)")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, radars, labels in train_loader:
            images, radars, labels = images.to(device), radars.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, radars) # [Batch, 4, 4, 5]
            
            # --- 복합 손실 계산 ---
            # 1. Objectness Loss (전체 그리드에 대해 존재 확률 계산)
            loss_obj = criterion_obj(outputs[..., 0], labels[..., 0])
            
            # 2. Box Loss (실제 물체가 있는 칸[labels[...,0] == 1]에 대해서만 계산)
            mask = labels[..., 0] == 1
            if mask.any():
                loss_box = criterion_box(outputs[mask][:, 1:], labels[mask][:, 1:])
            else:
                loss_box = 0
                
            loss = loss_obj + 5.0 * loss_box # 위치 오차에 더 큰 가중치(5.0) 부여
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증 단계
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, radars, labels in val_loader:
                images, radars, labels = images.to(device), radars.to(device), labels.to(device)
                outputs = model(images, radars)
                
                loss_obj = criterion_obj(outputs[..., 0], labels[..., 0])
                mask = labels[..., 0] == 1
                loss_box = criterion_box(outputs[mask][:, 1:], labels[mask][:, 1:]) if mask.any() else 0
                
                val_loss += (loss_obj + 5.0 * loss_box).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1:02d}/{epochs}] - Loss: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  🌟 다중 객체 모델 저장됨!")

if __name__ == "__main__":
    train()