# experiments/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 꿀팁: 상위 폴더(src)의 모듈을 에러 없이 불러오기 위한 절대 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

# 우리가 만든 모듈들 불러오기
from src.dataset import RadarVisionDataset
from src.models.attention_fusion import MidLevelAttentionFusion

def train():
    print("🚀 [Step 1] 학습 환경 설정 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 장치: {device}")

    # 데이터 로드
    data_dir = os.path.join(project_root, 'data', 'raw')
    dataset = RadarVisionDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"✅ 총 {len(dataset)}개의 데이터를 성공적으로 불러왔습니다.")

    print("🧠 [Step 2] 모델 및 최적화 도구 준비 중...")
    model = MidLevelAttentionFusion().to(device)
    criterion = nn.MSELoss() # 객체 탐지 박스 회귀를 위한 평균 제곱 오차
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    print(f"🔥 [Step 3] 본격적인 학습 시작! (총 {epochs} Epochs)")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, radars, labels) in enumerate(dataloader):
            # 데이터를 GPU 또는 CPU로 전송
            images = images.to(device)
            radars = radars.to(device)
            labels = labels.to(device)
            
            # 1. 예측 (Forward)
            outputs = model(images, radars)
            
            # 2. 오차 계산 (Loss)
            loss = criterion(outputs, labels)
            
            # 3. 학습 (Backward & Optimize)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 에폭마다 평균 Loss 출력 (이 값이 떨어져야 학습이 잘 되는 것입니다!)
        avg_loss = running_loss / len(dataloader)
        print(f"📈 Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("🎉 딥러닝 모델 학습이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    train()