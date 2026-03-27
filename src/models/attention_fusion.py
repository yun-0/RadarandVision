import torch
import torch.nn as nn

# --- [부품] Spatial Attention Module ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))

# --- [메인 모델] Hybrid Advanced Fusion Model ---
class AdvancedFusionModel(nn.Module):
    def __init__(self, mode='hybrid', grid_size=4):
        super().__init__()
        self.mode = mode.lower().strip()
        self.grid_size = grid_size
        
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, padding=1),
                nn.BatchNorm2d(out_f), 
                nn.LeakyReLU(0.1), 
                nn.MaxPool2d(2)
            )

        # 1. Feature Extractors (카메라 & 레이더)
        self.v_layer1 = conv_block(3, 16)  # 64 -> 32
        self.v_layer2 = conv_block(16, 32) # 32 -> 16
        self.v_layer3 = conv_block(32, 64) # 16 -> 8

        self.r_layer1 = conv_block(1, 16)
        self.r_layer2 = conv_block(16, 32)
        self.r_layer3 = conv_block(32, 64)

        # 2. Fusion Layers (모드별 설정)
        if self.mode == 'spatial':
            self.spatial_attn = SpatialAttention()
            self.fusion_conv = nn.Conv2d(128, 128, 1)
        elif self.mode == 'multiscale' or self.mode == 'hybrid':
            self.ms_fusion = nn.Conv2d(192, 128, 1) # v2, r2(32+32) + v3, r3(64+64) = 192
            if self.mode == 'hybrid':
                self.spatial_attn = SpatialAttention()

        # 3. Detection Head (4x4 그리드 출력)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, self.grid_size * self.grid_size * 5),
            nn.Sigmoid() # Confidence와 위치 좌표 모두 0~1 사이로
        )

    def forward(self, img, radar):
        # [중요] 여기서 'radar' 인자를 정상적으로 받아야 에러가 안 납니다!
        
        # 특징 추출
        v1 = self.v_layer1(img); v2 = self.v_layer2(v1); v3 = self.v_layer3(v2)
        r1 = self.r_layer1(radar); r2 = self.r_layer2(r1); r3 = self.r_layer3(r2)

        if self.mode == 'spatial':
            fused = torch.cat([v3, r3], dim=1)
            fused = fused * self.spatial_attn(fused)
            fused = self.fusion_conv(fused)
        
        elif self.mode == 'multiscale':
            v2_d = nn.functional.max_pool2d(v2, 2) # 16x16 -> 8x8
            r2_d = nn.functional.max_pool2d(r2, 2)
            fused = self.ms_fusion(torch.cat([v2_d, r2_d, v3, r3], dim=1))
            
        elif self.mode == 'hybrid':
            v2_d = nn.functional.max_pool2d(v2, 2)
            r2_d = nn.functional.max_pool2d(r2, 2)
            combined = torch.cat([v2_d, r2_d, v3, r3], dim=1)
            fused = self.ms_fusion(combined)
            fused = fused * self.spatial_attn(fused)
        
        else:
            fused = torch.cat([v3, r3], dim=1)

        # 회귀 및 그리드 형태로 변환 [Batch, 4, 4, 5]
        x = self.regressor(fused)
        return x.view(-1, self.grid_size, self.grid_size, 5)