# experiments/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # random_split 추가!

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import AdvancedFusionModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 장치: {device}")

    # 1. 데이터 불러오기 및 8:2로 쪼개기
    data_dir = os.path.join(project_root, 'data', 'raw')
    full_dataset = RadarVisionDataset(data_dir)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) # 시험은 섞을 필요 없음
    
    print(f"✅ 데이터 준비 완료: 총 {len(full_dataset)}개 (학습 {train_size}개 / 검증 {val_size}개)")

    # train.py 수정
    model = AdvancedFusionModel(mode='multiscale').to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 50 
    best_val_loss = float('inf') # 최고 기록을 저장할 변수 (처음엔 무한대로 설정)
    save_path = os.path.join(project_root, 'experiments', 'model_multiscale.pth')

    print(f"🔥 [정석 모드] Train / Validation 학습 시작! (총 {epochs} Epochs)\n")
    
    for epoch in range(epochs):
        # ==========================================
        # 📚 1. 학습 (Training) Phase
        # ==========================================
        model.train() # 학습 모드 ON (Dropout 작동)
        train_loss = 0.0
        
        for images, radars, labels in train_loader:
            images, radars = images.to(device), radars.to(device)
            labels = labels.squeeze().to(device) # 마법의 squeeze!
            
            optimizer.zero_grad()
            outputs = model(images, radars)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # ==========================================
        # 📝 2. 검증 (Validation) Phase
        # ==========================================
        model.eval() # 평가 모드 ON (Dropout 중지, 뇌세포 100% 가동)
        val_loss = 0.0
        
        with torch.no_grad(): # 역전파(학습) 금지!
            for images, radars, labels in val_loader:
                images, radars = images.to(device), radars.to(device)
                labels = labels.squeeze().to(device)
                
                outputs = model(images, radars)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        # ==========================================
        # 🏆 3. 결과 출력 및 최고 모델 저장
        # ==========================================
        print(f"Epoch [{epoch+1:02d}/{epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # 만약 이번 시험 점수(Val Loss)가 역대급으로 낮다면? (최고 기록 갱신)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  🌟 최고 기록 갱신! 모델 가중치 저장됨 (Val Loss: {best_val_loss:.6f})")

    print("\n🎉 모든 학습 및 검증이 완료되었습니다!")
    print(f"💾 최종적으로 가장 똑똑한 모델이 저장된 위치: {save_path}")

if __name__ == "__main__":
    train()