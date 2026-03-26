# src/models/attention_fusion.py
import torch
import torch.nn as nn

# [부품] Spatial Attention Module
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

# [최종 진화형] Hybrid Fusion Model
class AdvancedFusionModel(nn.Module):
    def __init__(self, mode='hybrid'): # 'spatial', 'multiscale', 'hybrid' 선택 가능
        super().__init__()
        self.mode = mode
        
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, padding=1),
                nn.BatchNorm2d(out_f), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
            )

        # 1. Feature Extractors
        self.v_layer1 = conv_block(3, 16)
        self.v_layer2 = conv_block(16, 32)
        self.v_layer3 = conv_block(32, 64)

        self.r_layer1 = conv_block(1, 16)
        self.r_layer2 = conv_block(16, 32)
        self.r_layer3 = conv_block(32, 64)

        # 2. Fusion Layers
        if mode == 'spatial':
            self.fusion_conv = nn.Conv2d(128, 128, 1)
            self.spatial_attn = SpatialAttention()
        elif mode == 'multiscale':
            self.ms_fusion = nn.Conv2d(192, 128, 1) # 32+32+64+64 = 192
        elif mode == 'hybrid':
            # [🔥 Hybrid] 멀티 스케일로 합친 후 + 공간 어텐션 적용!
            self.ms_fusion = nn.Conv2d(192, 128, 1)
            self.spatial_attn = SpatialAttention()

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.LeakyReLU(0.1),
            nn.Linear(256, 4), nn.Sigmoid()
        )

    def forward(self, img, radar):
        # 특징 추출
        v1 = self.v_layer1(img); v2 = self.v_layer2(v1); v3 = self.v_layer3(v2)
        r1 = self.r_layer1(radar); r2 = self.r_layer2(r1); r3 = self.r_layer3(r2)

        if self.mode == 'spatial':
            fused = torch.cat([v3, r3], dim=1)
            fused = fused * self.spatial_attn(fused)
        
        elif self.mode == 'multiscale':
            v2_d = nn.functional.max_pool2d(v2, 2)
            r2_d = nn.functional.max_pool2d(r2, 2)
            fused = self.ms_fusion(torch.cat([v2_d, r2_d, v3, r3], dim=1))
            
        elif self.mode == 'hybrid':
            # 1. 멀티 스케일 정보를 먼저 모읍니다 (돋보기)
            v2_d = nn.functional.max_pool2d(v2, 2)
            r2_d = nn.functional.max_pool2d(r2, 2)
            combined = torch.cat([v2_d, r2_d, v3, r3], dim=1)
            fused = self.ms_fusion(combined)
            # 2. 모인 정보에서 중요한 위치를 강조합니다 (손전등)
            fused = fused * self.spatial_attn(fused)

        return self.regressor(fused)