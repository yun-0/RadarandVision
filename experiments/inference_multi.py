# experiments/inference_multi.py
import os
import sys
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(project_root)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import AdvancedFusionModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = 4
    # 0.5(50%) 이상의 확신이 있는 박스만 화면에 표시합니다.
    threshold = 0.5 

    # 1. 모델 로드 (hybrid + grid_size 4)
    model = AdvancedFusionModel(mode='hybrid', grid_size=grid_size).to(device)
    weight_path = os.path.join(project_root, 'experiments', 'model_multi_best.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 2. 데이터 로드 (raw_multi 폴더!)
    data_dir = os.path.join(project_root, 'data', 'raw_multi')
    dataset = RadarVisionDataset(data_dir, grid_size=grid_size)
    
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, radar_tensor, label_tensor = dataset[idx]

    # 3. 예측
    with torch.no_grad():
        img_in = img_tensor.unsqueeze(0).to(device)
        rad_in = radar_tensor.unsqueeze(0).to(device)
        # output shape: [1, 4, 4, 5]
        pred = model(img_in, rad_in).squeeze(0).cpu()

    # 4. 시각화
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    img_display = img_tensor.permute(1, 2, 0).numpy()
    radar_display = radar_tensor.squeeze().numpy()

    ax[0].imshow(img_display)
    ax[0].set_title("Vision Multi-Detection")
    ax[1].imshow(radar_display, cmap='jet')
    ax[1].set_title("Radar Heatmap")

    # [핵심] 그리드를 돌면서 확률이 높은 박스만 그리기
    for i in range(grid_size):
        for j in range(grid_size):
            conf = pred[i, j, 0].item()
            if conf > threshold:
                cx, cy, w, h = pred[i, j, 1:]
                # 64x64 크기로 복원
                cx, cy, w, h = cx*64, cy*64, w*64, h*64
                rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, 
                                        linewidth=2, edgecolor='red', facecolor='none',
                                        label=f'AI ({conf:.2f})')
                ax[0].add_patch(rect)
                # 확률값 텍스트 표시
                ax[0].text(cx-w/2, cy-h/2-2, f'{conf:.2f}', color='red', fontsize=10, fontweight='bold')

    # 실제 정답(Ground Truth) 표시 (초록색)
    for i in range(grid_size):
        for j in range(grid_size):
            if label_tensor[i, j, 0] == 1:
                cx, cy, w, h = label_tensor[i, j, 1:]*64
                rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, 
                                        linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                ax[0].add_patch(rect)

    plt.suptitle(f"Multi-Object Detection (Threshold: {threshold})", fontsize=16)
    save_path = os.path.join(project_root, 'experiments', 'result_multi.png')
    plt.savefig(save_path)
    print(f"📸 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    main()