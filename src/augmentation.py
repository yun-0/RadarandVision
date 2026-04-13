"""
src/augmentation.py — 레이더-비전 쌍 데이터 증강 모듈

원칙:
  - 기하학적 변환(flip)은 img·radar 동시 적용 + bbox 좌표 변환
  - 광학적 변환(밝기/대비)은 비전(img)만 적용 — 레이더는 전자기 신호라 무관
  - 레이더 노이즈는 레이더만 적용 — 실제 다중경로(multipath)/클러터 변동 시뮬레이션

적용 금지:
  - Random Crop: 64×64 해상도에서 bbox 소실 위험
  - Rotation: bbox affine 변환이 복잡하고 불안정
"""

import random
import torch
import numpy as np


class RadarVisionAugment:
    """
    레이더-비전 쌍 증강 클래스.

    Args:
        flip_p    (float): 수평 플립 확률 (기본 0.5)
        vflip_p   (float): 수직 플립 확률 (기본 0.3)
        jitter_p  (float): 비전 밝기/대비 지터 확률 (기본 0.4)
        noise_p   (float): 레이더 가우시안 노이즈 확률 (기본 0.4)
        noise_std (float): 레이더 노이즈 표준편차 (기본 0.05)

    Input format:
        img   : torch.Tensor (3, H, W), 값 범위 [0, 1]
        radar : torch.Tensor (1, H, W), 값 범위 [0, 1]
        bboxes: numpy.ndarray (N, 4),   [cx, cy, w, h] normalized [0, 1]

    Returns:
        (img, radar, bboxes) — 동일 형태 반환
    """

    def __init__(self, flip_p=0.5, vflip_p=0.3, jitter_p=0.4,
                 noise_p=0.4, noise_std=0.05):
        self.flip_p    = flip_p
        self.vflip_p   = vflip_p
        self.jitter_p  = jitter_p
        self.noise_p   = noise_p
        self.noise_std = noise_std

    def __call__(self, img, radar, bboxes):
        """
        Args:
            img   : torch.Tensor (3, H, W)
            radar : torch.Tensor (1, H, W)
            bboxes: numpy.ndarray (N, 4) — [cx, cy, w, h]
        Returns:
            (img, radar, bboxes)
        """
        bboxes = bboxes.copy()  # 원본 훼손 방지

        # 1. 수평 플립 (Horizontal Flip)
        #    img/radar: dim=2(W) 기준 뒤집기
        #    bbox: cx → 1 - cx
        if random.random() < self.flip_p:
            img   = torch.flip(img,   dims=[2])
            radar = torch.flip(radar, dims=[2])
            bboxes[:, 0] = 1.0 - bboxes[:, 0]

        # 2. 수직 플립 (Vertical Flip)
        #    img/radar: dim=1(H) 기준 뒤집기
        #    bbox: cy → 1 - cy
        if random.random() < self.vflip_p:
            img   = torch.flip(img,   dims=[1])
            radar = torch.flip(radar, dims=[1])
            bboxes[:, 1] = 1.0 - bboxes[:, 1]

        # 3. 비전 밝기/대비 지터 (Vision-only)
        #    조도 환경 변화 및 카메라 노출 설정 변동 시뮬레이션
        if random.random() < self.jitter_p:
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            img = torch.clamp(img * contrast + (brightness - 1.0), 0.0, 1.0)

        # 4. 레이더 가우시안 노이즈 (Radar-only)
        #    다중경로(multipath) 반사 및 환경 클러터 변동 시뮬레이션
        #    참고: RCS (Radar Cross-Section) 변동은 일반적으로 0~10% 수준
        if random.random() < self.noise_p:
            noise = torch.randn_like(radar) * self.noise_std
            radar = torch.clamp(radar + noise, 0.0, 1.0)

        return img, radar, bboxes
