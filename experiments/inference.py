# experiments/inference.py
import os
import sys
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import MidLevelAttentionFusion

def draw_bbox(ax, bbox, color, label):
    # bbox는 [cx, cy, w, h] (0~1로 정규화된 값)
    # 이미지 크기가 64x64이므로 픽셀 단위로 복원합니다.
    img_size = 64
    cx, cy, w, h = bbox
    cx, cy, w, h = cx * img_size, cy * img_size, w * img_size, h * img_size
    
    # cx, cy (중심) -> x, y (좌측 상단 모서리)로 변환
    x = cx - w / 2
    y = cy - h / 2
    
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', label=label)
    ax.add_patch(rect)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 최고 성능의 뇌(가중치) 불러오기
    model = MidLevelAttentionFusion().to(device)
    weight_path = os.path.join(project_root, 'experiments', 'fusion_model_best.pth')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval() # 평가 모드 (Dropout 끄기)
    print(f"🧠 [완료] 최고의 모델({weight_path})을 성공적으로 뇌에 이식했습니다!")

    # 2. 데이터 불러오기
    data_dir = os.path.join(project_root, 'data', 'raw')
    dataset = RadarVisionDataset(data_dir)
    
    # 무작위로 1개의 데이터 뽑기
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, radar_tensor, true_bbox = dataset[idx]
    true_bbox = true_bbox.squeeze() # [1, 4] -> [4]

    # 3. 모델에게 예측시키기 (추론)
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(device)     # [1, 3, 64, 64]로 껍데기 씌움
        radar_input = radar_tensor.unsqueeze(0).to(device) # [1, 1, 64, 64]
        
        pred_bbox = model(img_input, radar_input)
        pred_bbox = pred_bbox.squeeze().cpu() # 예측 결과 [4]

    # 4. 결과 시각화 (그리기)
    img_display = img_tensor.permute(1, 2, 0).numpy() # [3, 64, 64] -> [64, 64, 3] 화면 출력용
    radar_display = radar_tensor.squeeze().numpy()    # [1, 64, 64] -> [64, 64]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # [왼쪽] 카메라 이미지 + BBox
    axes[0].imshow(img_display)
    axes[0].set_title("Vision (Camera)")
    draw_bbox(axes[0], true_bbox.numpy(), color='lime', label='Ground Truth (Real)')
    draw_bbox(axes[0], pred_bbox.numpy(), color='red', label='AI Prediction')
    axes[0].legend(loc='upper right')
    
    # [오른쪽] 레이더 이미지
    axes[1].imshow(radar_display, cmap='jet')
    axes[1].set_title("Radar (Heatmap)")
    
    plt.suptitle("AI Multi-Modal Sensor Fusion Prediction", fontsize=16)
    
    # 화면에 띄우거나 파일로 저장
    save_img_path = os.path.join(project_root, 'experiments', 'result_visual.png')
    plt.savefig(save_img_path)
    print(f"📸 [완료] AI가 예측한 결과 이미지가 저장되었습니다: {save_img_path}")
    
    # plt.show() # 서버 환경이 아니면 이 주석을 풀어서 바로 창을 띄울 수도 있습니다.

if __name__ == "__main__":
    main()