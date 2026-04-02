# experiments/train_v2.py
"""
train_v2.py — RadarVision 통합 학습 스크립트 (8×8 그리드 최종판)
─────────────────────────────────────────────────────────────────
기존 train.py / train_multi.py의 장점을 통합하고 다음을 개선:
  - grid_size=8 (4×4 → 8×8 전환)
  - 데이터: data/raw_multi
  - Focal Loss 적용 (Background 셀 압도 문제 해결)
  - Warm-up + Cosine Annealing LR 스케줄러
  - Gradient Clipping (학습 안정화)
  - MPS(Apple Silicon) 디바이스 지원
  - 학습 히스토리 저장 (Loss 곡선 분석용)

[실행]
  cd /Users/.../RadarandVision
  python experiments/train_v2.py

[Colab 실행]
  USE_COLAB = True 로 변경 후
  COLAB_DATA_DIR 경로 수정
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import math
import time
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ── 프로젝트 루트 경로 설정 ──────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import AdvancedFusionModel

# ══════════════════════════════════════════════════════════════════
# [설정] 여기만 수정하면 됩니다
# ══════════════════════════════════════════════════════════════════

USE_COLAB      = False
COLAB_DATA_DIR = '/content/drive/MyDrive/RadarandVision/data/raw_multi'
LOCAL_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw_multi')

CONFIG = {
    # 데이터
    'data_dir'      : COLAB_DATA_DIR if USE_COLAB else LOCAL_DATA_DIR,
    'grid_size'     : 8,        # ★ 핵심 변경: 4 → 8
    'img_size'      : 64,
    'val_ratio'     : 0.2,
    'num_workers'   : 0,        # Mac/Colab CPU 환경에서는 0 권장

    # 모델
    'fusion_mode'   : 'hybrid', # 'spatial' | 'multiscale' | 'hybrid'

    # 학습
    'batch_size'    : 16,
    'total_epochs'  : 100,
    'warmup_epochs' : 10,       # Warm-up 구간 (Loss 폭발 방지)
    'lr'            : 1e-4,     # Warm-up 목표 LR
    'lr_min'        : 1e-6,     # Cosine 최저 LR
    'weight_decay'  : 1e-4,

    # Loss 가중치
    'lambda_obj'    : 5.0,      # 객체 있는 셀 Confidence 강조
    'lambda_noobj'  : 0.5,      # 배경 셀(~59개) 억제 ★ 8×8 핵심
    'lambda_coord'  : 5.0,      # 좌표 회귀 (기존 train_multi.py 값 유지)

    # Focal Loss 파라미터
    'focal_alpha'   : 0.25,
    'focal_gamma'   : 2.0,

    # 저장
    'save_dir'      : os.path.join(ROOT_DIR, 'experiments', 'checkpoints_v2'),
    'save_interval' : 10,
    'log_interval'  : 5,
}

# ══════════════════════════════════════════════════════════════════
# Loss 함수
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    8×8 그리드에서 배경 셀(최대 63개)이 압도하는 문제를 완화.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred, target: 이미 Sigmoid 통과한 값 (0~1)
        bce    = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt     = torch.where(target == 1, pred, 1 - pred)
        weight = self.alpha * (1 - pt) ** self.gamma
        return (weight * bce).mean()


class FusionLoss(nn.Module):
    """
    멀티모달 탐지 통합 Loss.

    구성:
      - conf_obj   : 객체 있는 셀 → Focal Loss (강조)
      - conf_noobj : 배경 셀      → BCE (억제)
      - coord      : 좌표 회귀    → MSE (객체 셀만)
    """
    def __init__(self, cfg):
        super().__init__()
        self.focal       = FocalLoss(cfg['focal_alpha'], cfg['focal_gamma'])
        self.mse         = nn.MSELoss()
        self.bce         = nn.BCELoss()
        self.lam_obj     = cfg['lambda_obj']
        self.lam_noobj   = cfg['lambda_noobj']
        self.lam_coord   = cfg['lambda_coord']

    def forward(self, pred, target):
        """
        Args:
            pred   (Tensor): (B, G, G, 5) — 모델 출력
            target (Tensor): (B, G, G, 5) — 라벨
        Returns:
            total     (Tensor): 스칼라 Loss
            loss_dict (dict)  : 항목별 Loss (로그용)
        """
        obj_mask   = target[..., 0] == 1   # 객체 있는 셀
        noobj_mask = ~obj_mask              # 배경 셀

        # ── Confidence Loss ──────────────────────────────────────
        if obj_mask.any():
            conf_obj = self.focal(
                pred[..., 0][obj_mask],
                target[..., 0][obj_mask]
            )
        else:
            conf_obj = torch.tensor(0.0, device=pred.device)

        if noobj_mask.any():
            conf_noobj = self.bce(
                pred[..., 0][noobj_mask],
                target[..., 0][noobj_mask]
            )
        else:
            conf_noobj = torch.tensor(0.0, device=pred.device)

        # ── Coordinate Loss (객체 셀만) ──────────────────────────
        if obj_mask.any():
            coord_loss = self.mse(
                pred[..., 1:][obj_mask],
                target[..., 1:][obj_mask]
            )
        else:
            coord_loss = torch.tensor(0.0, device=pred.device)

        # ── 통합 ─────────────────────────────────────────────────
        total = (self.lam_obj   * conf_obj
               + self.lam_noobj * conf_noobj
               + self.lam_coord * coord_loss)

        return total, {
            'conf_obj'  : conf_obj.item(),
            'conf_noobj': conf_noobj.item(),
            'coord'     : coord_loss.item(),
            'total'     : total.item(),
        }


# ══════════════════════════════════════════════════════════════════
# LR 스케줄러
# ══════════════════════════════════════════════════════════════════

def get_lr(epoch, cfg):
    """
    Linear Warm-up → Cosine Annealing.
      - epoch  < warmup_epochs : 0 → lr 선형 증가
      - epoch >= warmup_epochs : lr → lr_min Cosine 감소
    """
    w     = cfg['warmup_epochs']
    T     = cfg['total_epochs'] - w
    lr    = cfg['lr']
    lr_min = cfg['lr_min']

    if epoch < w:
        return lr * (epoch + 1) / w
    t = epoch - w
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t / T))


# ══════════════════════════════════════════════════════════════════
# 학습 / 검증 루프
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    accum = {'conf_obj': 0, 'conf_noobj': 0, 'coord': 0, 'total': 0}

    for images, radars, labels in loader:
        images  = images.to(device)
        radars  = radars.to(device)
        labels  = labels.to(device)   # (B, G, G, 5) — squeeze 제거 ★

        optimizer.zero_grad()
        outputs = model(images, radars)          # (B, G, G, 5)
        loss, loss_dict = criterion(outputs, labels)
        loss.backward()

        # Gradient Clipping — 8×8 초기 학습 안정화
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in accum:
            accum[k] += loss_dict[k]

    n = len(loader)
    return {k: v / n for k, v in accum.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    accum = {'conf_obj': 0, 'conf_noobj': 0, 'coord': 0, 'total': 0}

    for images, radars, labels in loader:
        images  = images.to(device)
        radars  = radars.to(device)
        labels  = labels.to(device)

        outputs = model(images, radars)
        _, loss_dict = criterion(outputs, labels)

        for k in accum:
            accum[k] += loss_dict[k]

    n = len(loader)
    return {k: v / n for k, v in accum.items()}


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG

    # ── 디바이스 ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():   # Apple Silicon
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 62)
    print("  🚀 RadarVision train_v2.py 학습 시작")
    print("=" * 62)
    print(f"  디바이스    : {device}")
    print(f"  데이터 경로 : {cfg['data_dir']}")
    print(f"  그리드 크기 : {cfg['grid_size']}×{cfg['grid_size']}  (★ 8×8)")
    print(f"  Fusion 모드 : {cfg['fusion_mode']}")
    print(f"  총 에폭    : {cfg['total_epochs']}  (Warm-up: {cfg['warmup_epochs']})")
    print(f"  배치 크기  : {cfg['batch_size']}")
    print(f"  Loss 가중치 : obj={cfg['lambda_obj']} / noobj={cfg['lambda_noobj']} / coord={cfg['lambda_coord']}")
    print("=" * 62)

    # ── 데이터셋 ─────────────────────────────────────────────────
    full_dataset = RadarVisionDataset(
        data_dir  = cfg['data_dir'],
        grid_size = cfg['grid_size'],
        img_size  = cfg['img_size'],
    )

    val_size   = int(len(full_dataset) * cfg['val_ratio'])
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=cfg['num_workers'])
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'])

    print(f"  학습 샘플  : {train_size}개  |  검증 샘플: {val_size}개\n")

    # ── 모델 ─────────────────────────────────────────────────────
    model = AdvancedFusionModel(
        mode      = cfg['fusion_mode'],
        grid_size = cfg['grid_size'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  모델 파라미터: {n_params:,}개\n")

    # ── Loss / Optimizer ─────────────────────────────────────────
    criterion = FusionLoss(cfg)
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg['lr_min'],   # Warm-up 시작은 낮게
        weight_decay = cfg['weight_decay']
    )

    # ── 저장 폴더 ────────────────────────────────────────────────
    os.makedirs(cfg['save_dir'], exist_ok=True)

    # ── 학습 기록 ────────────────────────────────────────────────
    history = {
        'train_total' : [], 'val_total'    : [],
        'train_obj'   : [], 'train_noobj'  : [], 'train_coord': [],
        'val_obj'     : [], 'val_noobj'    : [], 'val_coord'  : [],
        'lr'          : [],
    }
    best_val_loss = float('inf')

    # ── 학습 루프 ────────────────────────────────────────────────
    for epoch in range(cfg['total_epochs']):
        t0 = time.time()

        # LR 업데이트
        target_lr = get_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = target_lr

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)

        # 기록
        history['train_total'].append(train_loss['total'])
        history['val_total'].append(val_loss['total'])
        history['train_obj'].append(train_loss['conf_obj'])
        history['train_noobj'].append(train_loss['conf_noobj'])
        history['train_coord'].append(train_loss['coord'])
        history['val_obj'].append(val_loss['conf_obj'])
        history['val_noobj'].append(val_loss['conf_noobj'])
        history['val_coord'].append(val_loss['coord'])
        history['lr'].append(target_lr)

        elapsed = time.time() - t0
        phase   = "Warm-up" if epoch < cfg['warmup_epochs'] else "Cosine "

        # 로그 출력
        if (epoch + 1) % cfg['log_interval'] == 0 or epoch == 0:
            print(
                f"  [{phase}] Ep {epoch+1:>3}/{cfg['total_epochs']}  "
                f"LR={target_lr:.1e}  "
                f"Train={train_loss['total']:.4f} "
                f"(obj={train_loss['conf_obj']:.3f} "
                f"noobj={train_loss['conf_noobj']:.3f} "
                f"coord={train_loss['coord']:.3f})  "
                f"Val={val_loss['total']:.4f}  "
                f"[{elapsed:.1f}s]"
            )

        # 🌟 Best 모델 저장
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                'epoch'          : epoch + 1,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss'       : best_val_loss,
                'config'         : cfg,
            }, os.path.join(cfg['save_dir'], 'best_model_v2.pth'))
            print(f"  🌟 Best 갱신! Val Loss: {best_val_loss:.4f} → 저장 완료")

        # 주기적 체크포인트
        if (epoch + 1) % cfg['save_interval'] == 0:
            torch.save({
                'epoch'      : epoch + 1,
                'model_state': model.state_dict(),
                'val_loss'   : val_loss['total'],
                'history'    : history,
                'config'     : cfg,
            }, os.path.join(cfg['save_dir'], f'ckpt_ep{epoch+1}.pth'))

    # ── 완료 ─────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  🎉 학습 완료!  Best Val Loss: {best_val_loss:.4f}")
    print(f"  저장 위치: {cfg['save_dir']}")
    print("=" * 62)

    # Occlusion 통계
    stats = full_dataset.get_occlusion_stats()
    print(f"\n  [Occlusion 통계]")
    print(f"  전체 객체: {stats['total']}  충돌: {stats['occluded']}  비율: {stats['rate']*100:.2f}%")

    # 히스토리 저장
    np.save(os.path.join(cfg['save_dir'], 'history_v2.npy'), history)
    print(f"  히스토리 저장: experiments/checkpoints_v2/history_v2.npy")


if __name__ == '__main__':
    main()