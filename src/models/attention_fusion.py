import torch
import torch.nn as nn

# --- [부품] SE (Squeeze-and-Excitation) Channel Attention ---
class SEBlock(nn.Module):
    """
    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    채널별 중요도를 학습하여 퓨전된 특징의 품질을 향상.
    파라미터 수: 2 * (C * C/r) — reduction=8 기준 약 2K params
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * w


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
    def __init__(self, mode='hybrid', grid_size=8):
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
                self.se = SEBlock(128, reduction=8)  # 채널 어텐션 추가

        # 3. Detection Head — Conv 기반 (FC 헤드 대비 공간정보 보존)
        # 입력: (B, 128, G, G), 출력: (B, 5, G, G) → permute → (B, G, G, 5)
        # 파라미터: ~92K (기존 FC 헤드 ~4.4M 대비 50× 감소)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 5, kernel_size=1),  # 5 = [conf, cx, cy, w, h] per cell
            nn.Sigmoid(),
        )

    def forward(self, img, radar):
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
            fused = self.se(fused)                   # SE 채널 어텐션
            fused = fused * self.spatial_attn(fused) # Spatial 어텐션

        else:
            fused = torch.cat([v3, r3], dim=1)

        # Conv Detection Head — 공간 구조 보존
        x = self.head(fused)              # (B, 5, G, G)
        return x.permute(0, 2, 3, 1)     # (B, G, G, 5)
