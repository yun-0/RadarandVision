# experiments/train_v2.py
"""
train_v2.py — RadarVision 통합 학습 스크립트 (8×8 그리드 최종판 v2)
─────────────────────────────────────────────────────────────────
개선 사항 (v1 대비):
  [모델]
  - Conv 기반 Detection Head (FC 헤드 대비 50× 파라미터 감소, 공간정보 보존)
  - SE Block (채널 어텐션) 추가 — hybrid 모드

  [손실 함수]
  - 좌표 회귀: MSE → CIoU Loss (Zheng et al., AAAI 2020)
    → IoU 직접 최적화로 mAP와 손실 상관관계 향상

  [학습 효율]
  - AMP (Automatic Mixed Precision) — CUDA/Jetson에서 ~2× 속도
  - TensorBoard 로깅 — 논문용 Loss 곡선 및 mAP 그래프

  [평가]
  - mAP@0.5, mAP@0.5:0.95 (COCO) 매 5 에폭 측정 — IEEE 논문 정량 지표

  [데이터]
  - file_list 기반 train/val 분리 → 학습셋에만 데이터 증강 적용
  - RadarVisionAugment: H-Flip, V-Flip, Vision Jitter, Radar Noise

[실행]
  cd /Users/.../RadarandVision
  python experiments/train_v2.py

[TensorBoard]
  tensorboard --logdir experiments/checkpoints_v2/tb_logs/

[Colab]
  USE_COLAB = True 로 변경 후 COLAB_DATA_DIR 경로 수정
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
from torch.utils.data import DataLoader

# ── 프로젝트 루트 경로 설정 ──────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import AdvancedFusionModel
from src.augmentation import RadarVisionAugment
from src.metrics import decode_grid_to_boxes, compute_map, compute_map_range

# ══════════════════════════════════════════════════════════════════
# [설정] 여기만 수정하면 됩니다
# ══════════════════════════════════════════════════════════════════

USE_COLAB      = False
COLAB_DATA_DIR = '/content/drive/MyDrive/RadarandVision/data/raw_multi'
LOCAL_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw_multi')

CONFIG = {
    # 데이터
    'data_dir'      : COLAB_DATA_DIR if USE_COLAB else LOCAL_DATA_DIR,
    'grid_size'     : 8,
    'img_size'      : 64,
    'val_ratio'     : 0.2,
    'num_workers'   : 0,

    # 모델
    'fusion_mode'   : 'hybrid',

    # 학습
    'batch_size'    : 16,
    'total_epochs'  : 100,
    'warmup_epochs' : 10,
    'lr'            : 1e-4,
    'lr_min'        : 1e-6,
    'weight_decay'  : 1e-4,

    # Loss 가중치
    'lambda_obj'    : 5.0,
    'lambda_noobj'  : 0.5,
    'lambda_coord'  : 5.0,

    # Focal Loss 파라미터
    'focal_alpha'   : 0.25,
    'focal_gamma'   : 2.0,

    # AMP (CUDA 전용, MPS/CPU에서는 자동 비활성화)
    'use_amp'       : True,

    # mAP 평가 주기
    'map_eval_interval' : 5,
    'map_conf_threshold': 0.01,

    # 데이터 증강
    'aug_flip_p'    : 0.5,
    'aug_vflip_p'   : 0.3,
    'aug_jitter_p'  : 0.4,
    'aug_noise_p'   : 0.4,
    'aug_noise_std' : 0.05,

    # 저장
    'save_dir'      : os.path.join(ROOT_DIR, 'experiments', 'checkpoints_v2'),
    'save_interval' : 10,
    'log_interval'  : 5,
}

# ══════════════════════════════════════════════════════════════════
# CIoU Loss (좌표 회귀용)
# ══════════════════════════════════════════════════════════════════

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete-IoU Loss.
    참고: Zheng et al., "Enhancing Geometric Factors in Model Learning
          and Inference for Object Detection and Instance Segmentation",
          IEEE TCSVT 2021 (arXiv:2005.03572).

    기존 MSE 대비 장점:
      1) IoU 직접 최적화 → mAP와 손실 상관관계 향상
      2) 중심점 거리 패널티 → 예측 박스가 GT 방향으로 수렴
      3) 종횡비 일관성 항 → w/h 비율 정확도 향상

    Args:
        pred_boxes   : (N, 4) Tensor — [cx, cy, w, h] normalized [0,1]
        target_boxes : (N, 4) Tensor — [cx, cy, w, h] normalized [0,1]

    Returns:
        Scalar — 평균 CIoU 손실 (범위: 0 이상)
    """
    # xyxy 변환
    p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    p_area = (p_x2 - p_x1).clamp(0) * (p_y2 - p_y1).clamp(0)
    t_area = (t_x2 - t_x1).clamp(0) * (t_y2 - t_y1).clamp(0)
    union  = p_area + t_area - inter + eps
    iou    = inter / union

    # 포함 박스(Enclosing box) 대각선 제곱
    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # 중심점 거리 제곱
    rho2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
           (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # 종횡비 일관성 항
    v = (4.0 / (math.pi ** 2)) * (
        torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + eps)) -
        torch.atan(pred_boxes[:, 2]   / (pred_boxes[:, 3]   + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1 - ciou).mean()


# ══════════════════════════════════════════════════════════════════
# Loss 함수
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    8×8 그리드에서 배경 셀(최대 63개)이 압도하는 클래스 불균형 문제 완화.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
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
      - coord      : 좌표 회귀    → CIoU Loss (객체 셀만)
    """
    def __init__(self, cfg):
        super().__init__()
        self.focal       = FocalLoss(cfg['focal_alpha'], cfg['focal_gamma'])
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
        obj_mask   = target[..., 0] == 1
        noobj_mask = ~obj_mask

        # ── Confidence Loss ──────────────────────────────────────
        if obj_mask.any():
            conf_obj = self.focal(pred[..., 0][obj_mask], target[..., 0][obj_mask])
        else:
            conf_obj = torch.tensor(0.0, device=pred.device)

        if noobj_mask.any():
            conf_noobj = self.bce(pred[..., 0][noobj_mask], target[..., 0][noobj_mask])
        else:
            conf_noobj = torch.tensor(0.0, device=pred.device)

        # ── Coordinate Loss — CIoU (객체 셀만) ──────────────────
        if obj_mask.any():
            coord_loss = ciou_loss(pred[..., 1:][obj_mask], target[..., 1:][obj_mask])
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
    """Linear Warm-up → Cosine Annealing."""
    w      = cfg['warmup_epochs']
    T      = cfg['total_epochs'] - w
    lr     = cfg['lr']
    lr_min = cfg['lr_min']

    if epoch < w:
        return lr * (epoch + 1) / w
    t = epoch - w
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t / T))


# ══════════════════════════════════════════════════════════════════
# 학습 / 검증 루프
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    accum = {'conf_obj': 0, 'conf_noobj': 0, 'coord': 0, 'total': 0}
    use_amp = scaler is not None

    for images, radars, labels in loader:
        images = images.to(device)
        radars = radars.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, radars)
            loss, loss_dict = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
        images = images.to(device)
        radars = radars.to(device)
        labels = labels.to(device)

        outputs = model(images, radars)
        _, loss_dict = criterion(outputs, labels)

        for k in accum:
            accum[k] += loss_dict[k]

    n = len(loader)
    return {k: v / n for k, v in accum.items()}


@torch.no_grad()
def evaluate_map(model, loader, device, conf_threshold=0.01):
    """
    검증셋 전체에 대해 mAP@0.5 및 mAP@0.5:0.95 계산.

    Args:
        model          : 학습된 모델
        loader         : 검증 DataLoader
        device         : torch.device
        conf_threshold : PR 곡선용 낮은 threshold (기본 0.01)

    Returns:
        (map50, map50_95): float, float
    """
    model.eval()
    all_preds = []
    all_gts   = []

    for images, radars, labels in loader:
        images = images.to(device)
        radars = radars.to(device)
        outputs = model(images, radars).cpu()  # (B, G, G, 5)

        for b in range(outputs.shape[0]):
            preds = decode_grid_to_boxes(outputs[b], conf_threshold)
            all_preds.append(preds)

            # GT 박스 추출
            G = labels.shape[1]
            gt_boxes = []
            for gy in range(G):
                for gx in range(G):
                    if labels[b, gy, gx, 0] == 1:
                        gt_boxes.append(labels[b, gy, gx, 1:].tolist())
            all_gts.append(gt_boxes)

    map50    = compute_map(all_preds, all_gts, iou_threshold=0.5)
    map50_95 = compute_map_range(all_preds, all_gts)
    return map50, map50_95


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG

    # ── 디바이스 ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    use_amp = cfg['use_amp'] and (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    print("=" * 62)
    print("  RadarVision train_v2.py 학습 시작 (v2 — Conv Head + CIoU)")
    print("=" * 62)
    print(f"  디바이스    : {device}  (AMP: {use_amp})")
    print(f"  데이터 경로 : {cfg['data_dir']}")
    print(f"  그리드 크기 : {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Fusion 모드 : {cfg['fusion_mode']}")
    print(f"  총 에폭    : {cfg['total_epochs']}  (Warm-up: {cfg['warmup_epochs']})")
    print(f"  배치 크기  : {cfg['batch_size']}")
    print(f"  Loss 가중치 : obj={cfg['lambda_obj']} / noobj={cfg['lambda_noobj']} / coord={cfg['lambda_coord']}")
    print("=" * 62)

    # ── 데이터셋 분리 (train/val 각각 별도 transform 적용) ──────
    all_files = sorted([f for f in os.listdir(cfg['data_dir']) if f.endswith('.mat')])
    rng       = np.random.default_rng(42)
    shuffled  = rng.permutation(all_files).tolist()
    n_val     = int(len(shuffled) * cfg['val_ratio'])
    val_files   = shuffled[:n_val]
    train_files = shuffled[n_val:]

    train_transform = RadarVisionAugment(
        flip_p    = cfg['aug_flip_p'],
        vflip_p   = cfg['aug_vflip_p'],
        jitter_p  = cfg['aug_jitter_p'],
        noise_p   = cfg['aug_noise_p'],
        noise_std = cfg['aug_noise_std'],
    )

    train_ds = RadarVisionDataset(
        data_dir  = cfg['data_dir'],
        grid_size = cfg['grid_size'],
        img_size  = cfg['img_size'],
        file_list = train_files,
        transform = train_transform,
    )
    val_ds = RadarVisionDataset(
        data_dir  = cfg['data_dir'],
        grid_size = cfg['grid_size'],
        img_size  = cfg['img_size'],
        file_list = val_files,
        transform = None,       # 검증셋은 증강 없음
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=cfg['num_workers'])
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'])

    print(f"  학습 샘플  : {len(train_files)}개  |  검증 샘플: {len(val_files)}개\n")

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
        lr           = cfg['lr_min'],
        weight_decay = cfg['weight_decay']
    )

    # ── 저장 폴더 / TensorBoard ──────────────────────────────────
    os.makedirs(cfg['save_dir'], exist_ok=True)
    tb_log_dir = os.path.join(cfg['save_dir'], 'tb_logs')

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tb_log_dir)
        use_tb = True
        print(f"  TensorBoard : {tb_log_dir}")
        print(f"  실행 명령  : tensorboard --logdir {tb_log_dir}\n")
    except ImportError:
        writer   = None
        use_tb   = False
        print("  [경고] tensorboard 미설치 — pip install tensorboard 권장\n")

    # ── 학습 기록 ────────────────────────────────────────────────
    history = {
        'train_total' : [], 'val_total'    : [],
        'train_obj'   : [], 'train_noobj'  : [], 'train_coord': [],
        'val_obj'     : [], 'val_noobj'    : [], 'val_coord'  : [],
        'lr'          : [], 'map50'        : [], 'map50_95'   : [],
    }
    best_val_loss = float('inf')

    # ── 학습 루프 ────────────────────────────────────────────────
    for epoch in range(cfg['total_epochs']):
        t0 = time.time()

        target_lr = get_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = target_lr

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
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

        # mAP 평가 (매 N 에폭)
        map50, map50_95 = 0.0, 0.0
        if (epoch + 1) % cfg['map_eval_interval'] == 0:
            map50, map50_95 = evaluate_map(
                model, val_loader, device, cfg['map_conf_threshold']
            )
            history['map50'].append(map50)
            history['map50_95'].append(map50_95)

        # TensorBoard 로깅
        if use_tb:
            writer.add_scalar('Loss/train_total',   train_loss['total'],      epoch)
            writer.add_scalar('Loss/train_obj',     train_loss['conf_obj'],   epoch)
            writer.add_scalar('Loss/train_noobj',   train_loss['conf_noobj'], epoch)
            writer.add_scalar('Loss/train_coord',   train_loss['coord'],      epoch)
            writer.add_scalar('Loss/val_total',     val_loss['total'],        epoch)
            writer.add_scalar('Loss/val_obj',       val_loss['conf_obj'],     epoch)
            writer.add_scalar('Loss/val_coord',     val_loss['coord'],        epoch)
            writer.add_scalar('LR',                 target_lr,                epoch)
            if (epoch + 1) % cfg['map_eval_interval'] == 0:
                writer.add_scalar('Metric/mAP50',    map50,    epoch)
                writer.add_scalar('Metric/mAP50_95', map50_95, epoch)

        elapsed = time.time() - t0
        phase   = "Warm-up" if epoch < cfg['warmup_epochs'] else "Cosine "

        # 콘솔 로그
        if (epoch + 1) % cfg['log_interval'] == 0 or epoch == 0:
            map_str = (f"  mAP@0.5={map50:.4f}  mAP@0.5:0.95={map50_95:.4f}"
                       if (epoch + 1) % cfg['map_eval_interval'] == 0 else "")
            print(
                f"  [{phase}] Ep {epoch+1:>3}/{cfg['total_epochs']}  "
                f"LR={target_lr:.1e}  "
                f"Train={train_loss['total']:.4f} "
                f"(obj={train_loss['conf_obj']:.3f} "
                f"noobj={train_loss['conf_noobj']:.3f} "
                f"coord={train_loss['coord']:.3f})  "
                f"Val={val_loss['total']:.4f}  "
                f"[{elapsed:.1f}s]{map_str}"
            )

        # Best 모델 저장
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                'epoch'          : epoch + 1,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss'       : best_val_loss,
                'config'         : cfg,
            }, os.path.join(cfg['save_dir'], 'best_model_v2.pth'))
            print(f"  [Best] Val Loss: {best_val_loss:.4f} 갱신 → 저장 완료")

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
    if use_tb:
        writer.close()

    print()
    print("=" * 62)
    print(f"  학습 완료!  Best Val Loss: {best_val_loss:.4f}")
    print(f"  저장 위치: {cfg['save_dir']}")
    print("=" * 62)

    # 최종 mAP 평가
    print("\n  [최종 mAP 평가 (전체 검증셋)]")
    final_map50, final_map5095 = evaluate_map(model, val_loader, device, cfg['map_conf_threshold'])
    print(f"  mAP@0.5       : {final_map50:.4f}")
    print(f"  mAP@0.5:0.95  : {final_map5095:.4f}")

    # Occlusion 통계
    stats = train_ds.get_occlusion_stats()
    print(f"\n  [Occlusion 통계 — 학습셋]")
    print(f"  전체 객체: {stats['total']}  충돌: {stats['occluded']}  비율: {stats['rate']*100:.2f}%")

    # 히스토리 저장
    np.save(os.path.join(cfg['save_dir'], 'history_v2.npy'), history)
    print(f"  히스토리: experiments/checkpoints_v2/history_v2.npy")


if __name__ == '__main__':
    main()
