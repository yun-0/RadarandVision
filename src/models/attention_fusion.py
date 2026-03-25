# src/models/attention_fusion.py
import torch
import torch.nn as nn

class MidLevelAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Vision Feature Extractor (RGB: 3채널 -> 32채널)
        self.vision_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        # 2. Radar Feature Extractor (Radar: 1채널 -> 32채널)
        self.radar_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        # 3. Attention Mechanism (융합된 64채널 특징 중 중요한 부분 강조)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), 
            nn.Sigmoid()
        )
        
        # 4. Bounding Box Regressor (최종 출력: cx, cy, w, h)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid() # 좌표값이 0~1 사이이므로 Sigmoid 사용
        )

    def forward(self, img, radar):
        v_feat = self.vision_net(img)
        r_feat = self.radar_net(radar)
        
        # Feature 단계에서 병합 (Mid-level Fusion)
        fused = torch.cat([v_feat, r_feat], dim=1) 
        
        # Attention 가중치 적용
        attn_weights = self.attention(fused)
        attended_fused = fused * attn_weights
        
        # BBox 예측
        bbox = self.regressor(attended_fused)
        return bbox