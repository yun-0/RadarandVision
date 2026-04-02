# experiments/inference_v2.py
"""
inference_v2.py — RadarVision 추론 스크립트 (8×8 그리드 최종판)
─────────────────────────────────────────────────────────────────
기존 inference.py / inference_multi.py를 통합하고 다음을 개선:
  - grid_size=8 (8×8 그리드)
  - best_model_v2.pth 자동 로드
  - 3패널 시각화: RGB | Radar | Confidence 히트맵
  - NMS(Non-Maximum Suppression) 적용 — 중복 박스 제거
  - Confidence 임계값 CLI 인자 지원

[실행]
  python experiments/inference_v2.py
  python experiments/inference_v2.py --threshold 0.4 --n 3
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import random
import argparse
import platform
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm

# ── 한글 폰트 설정 ───────────────────────────────────────────────
def set_korean_font():
    system = platform.system()
    candidates = {
        'Darwin' : ['/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                    '/Library/Fonts/NanumGothic.ttf'],
        'Windows': ['C:/Windows/Fonts/malgun.ttf'],
        'Linux'  : ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf'],
    }.get(system, [])
    for fpath in candidates:
        if os.path.exists(fpath):
            plt.rcParams['font.family'] = fm.FontProperties(fname=fpath).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return

set_korean_font()

# ── 프로젝트 경로 설정 ───────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.dataset import RadarVisionDataset
from src.models.attention_fusion import AdvancedFusionModel

# ══════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════

GRID_SIZE   = 8
IMG_SIZE    = 64
FUSION_MODE = 'hybrid'
WEIGHT_PATH = os.path.join(ROOT_DIR, 'experiments', 'checkpoints_v2', 'best_model_v2.pth')
DATA_DIR    = os.path.join(ROOT_DIR, 'data', 'raw_multi')
SAVE_DIR    = os.path.join(ROOT_DIR, 'experiments', 'results_v2')

# ══════════════════════════════════════════════════════════════════
# NMS (Non-Maximum Suppression)
# ══════════════════════════════════════════════════════════════════

def iou(box1, box2):
    """두 박스의 IoU 계산. 박스: [cx, cy, w, h] (픽셀 단위)"""
    def to_xyxy(b):
        return b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2

    x1,y1,x2,y2 = to_xyxy(box1)
    x3,y3,x4,y4 = to_xyxy(box2)

    ix1, iy1 = max(x1,x3), max(y1,y3)
    ix2, iy2 = min(x2,x4), min(y2,y4)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter
    return inter / (union + 1e-6)


def apply_nms(detections, iou_threshold=0.4):
    """
    Args:
        detections (list of dict): {'conf', 'cx', 'cy', 'w', 'h'}
    Returns:
        list of dict: NMS 통과한 박스만
    """
    if not detections:
        return []

    # Confidence 내림차순 정렬
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if iou(
                [best['cx'], best['cy'], best['w'], best['h']],
                [d['cx'],    d['cy'],    d['w'],    d['h']]
            ) < iou_threshold
        ]
    return kept


# ══════════════════════════════════════════════════════════════════
# 추론 + 시각화
# ══════════════════════════════════════════════════════════════════

def decode_predictions(pred, threshold, grid_size=GRID_SIZE, img_size=IMG_SIZE):
    """
    모델 출력(grid_size, grid_size, 5)에서 탐지 결과 추출 + NMS 적용.

    Args:
        pred      (Tensor): (G, G, 5), CPU
        threshold (float) : Confidence 임계값
    Returns:
        list of dict: NMS 통과한 탐지 결과
    """
    detections = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            conf = pred[gy, gx, 0].item()
            if conf > threshold:
                cx = pred[gy, gx, 1].item() * img_size
                cy = pred[gy, gx, 2].item() * img_size
                w  = pred[gy, gx, 3].item() * img_size
                h  = pred[gy, gx, 4].item() * img_size
                detections.append({'conf': conf, 'cx': cx, 'cy': cy, 'w': w, 'h': h})

    return apply_nms(detections)


def decode_gt(label, grid_size=GRID_SIZE, img_size=IMG_SIZE):
    """Ground Truth 라벨 디코딩."""
    gts = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            if label[gy, gx, 0] == 1:
                cx = label[gy, gx, 1].item() * img_size
                cy = label[gy, gx, 2].item() * img_size
                w  = label[gy, gx, 3].item() * img_size
                h  = label[gy, gx, 4].item() * img_size
                gts.append({'cx': cx, 'cy': cy, 'w': w, 'h': h})
    return gts


def draw_boxes(ax, boxes, color, linestyle='-', prefix='AI'):
    """Axes에 박스 목록 그리기."""
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i, b in enumerate(boxes):
        c  = colors[i % len(colors)] if color == 'auto' else color
        x  = b['cx'] - b['w'] / 2
        y  = b['cy'] - b['h'] / 2
        ax.add_patch(patches.Rectangle(
            (x, y), b['w'], b['h'],
            linewidth=2, edgecolor=c, facecolor='none', linestyle=linestyle
        ))
        label = f"{prefix} {b.get('conf', ''):.2f}" if 'conf' in b else prefix
        ax.text(x, y - 2, label, color=c, fontsize=8,
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.4, pad=1))


def build_heatmap(pred, grid_size=GRID_SIZE):
    """예측 Confidence를 히트맵 배열로 변환."""
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for gy in range(grid_size):
        for gx in range(grid_size):
            heatmap[gy, gx] = pred[gy, gx, 0].item()
    return heatmap


def run_inference(model, dataset, device, idx, threshold, save_dir, grid_size=GRID_SIZE):
    """단일 샘플 추론 및 시각화 저장."""
    img_tensor, radar_tensor, label_tensor = dataset[idx]
    fname = dataset.files[idx].replace('.mat', '')

    # 추론
    with torch.no_grad():
        pred = model(
            img_tensor.unsqueeze(0).to(device),
            radar_tensor.unsqueeze(0).to(device)
        ).squeeze(0).cpu()   # (G, G, 5)

    # 디코딩
    detections = decode_predictions(pred, threshold, grid_size)
    gts        = decode_gt(label_tensor, grid_size)

    print(f"  [{fname}]  GT: {len(gts)}개  탐지: {len(detections)}개  "
          f"(threshold={threshold})")

    # 이미지 데이터 준비
    img_np   = img_tensor.permute(1, 2, 0).numpy()
    img_np   = np.clip(img_np, 0, 1)
    radar_np = radar_tensor.squeeze().numpy()
    r_min, r_max = radar_np.min(), radar_np.max()
    radar_norm = (radar_np - r_min) / (r_max - r_min + 1e-8)
    heatmap  = build_heatmap(pred, grid_size)

    # ── 3패널 시각화 ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{fname}  |  GT: {len(gts)}개  |  탐지: {len(detections)}개  "
        f"(threshold={threshold})",
        fontsize=12, fontweight='bold'
    )

    # Panel 1: RGB + 예측 박스 + GT
    ax0 = axes[0]
    ax0.imshow(img_np)
    ax0.set_title('RGB Image', fontsize=11)
    draw_boxes(ax0, gts,        color='lime',  linestyle='--', prefix='GT')
    draw_boxes(ax0, detections, color='auto',  linestyle='-',  prefix='AI')
    ax0.axis('off')

    # Panel 2: Radar RD Map + GT 박스
    ax1 = axes[1]
    im1 = ax1.imshow(radar_norm, cmap='jet', vmin=0, vmax=1)
    ax1.set_title('Radar RD Map', fontsize=11)
    draw_boxes(ax1, gts,        color='white', linestyle='--', prefix='GT')
    draw_boxes(ax1, detections, color='red',   linestyle='-',  prefix='AI')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis('off')

    # Panel 3: Confidence 히트맵
    ax2 = axes[2]
    im2 = ax2.imshow(heatmap, cmap='hot', vmin=0, vmax=1,
                     interpolation='nearest', aspect='auto')
    ax2.set_title(f'Confidence Heatmap ({grid_size}×{grid_size})', fontsize=11)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 그리드 라인
    for v in range(grid_size + 1):
        ax2.axvline(v - 0.5, color='gray', linewidth=0.4, alpha=0.5)
        ax2.axhline(v - 0.5, color='gray', linewidth=0.4, alpha=0.5)

    # threshold 라인 표시
    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))
    ax2.set_xticklabels(range(grid_size), fontsize=7)
    ax2.set_yticklabels(range(grid_size), fontsize=7)

    plt.tight_layout()

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{fname}_result_v2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 저장: {save_path}")
    return len(gts), len(detections)


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='RadarVision 추론 v2')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence 임계값 (기본값: 0.5)')
    parser.add_argument('--n',         type=int,   default=5,
                        help='추론할 샘플 수 (기본값: 5, 0=전체)')
    parser.add_argument('--seed',      type=int,   default=None,
                        help='랜덤 시드 (재현성)')
    parser.add_argument('--weight',    type=str,   default=WEIGHT_PATH,
                        help='모델 가중치 경로')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── 디바이스 ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 55)
    print("  🔍 RadarVision inference_v2.py")
    print("=" * 55)
    print(f"  디바이스    : {device}")
    print(f"  가중치 경로 : {args.weight}")
    print(f"  Threshold   : {args.threshold}")
    print(f"  샘플 수     : {args.n if args.n > 0 else '전체'}")
    print("=" * 55)

    # ── 모델 로드 ────────────────────────────────────────────────
    model = AdvancedFusionModel(mode=FUSION_MODE, grid_size=GRID_SIZE).to(device)

    if not os.path.exists(args.weight):
        print(f"\n❌ 가중치 파일 없음: {args.weight}")
        print("   train_v2.py를 먼저 실행해 주세요.")
        return

    ckpt = torch.load(args.weight, map_location=device)
    # state_dict만 있는 경우와 dict 래핑된 경우 모두 처리
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()

    saved_epoch = ckpt.get('epoch', '?')
    saved_loss  = ckpt.get('val_loss', '?')
    print(f"\n  🧠 모델 로드 완료  (저장 에폭: {saved_epoch}  Val Loss: {saved_loss})")

    # ── 데이터셋 ─────────────────────────────────────────────────
    dataset = RadarVisionDataset(DATA_DIR, grid_size=GRID_SIZE, img_size=IMG_SIZE)

    n = args.n if args.n > 0 else len(dataset)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    print(f"\n  총 {len(indices)}개 샘플 추론 시작...\n")

    total_gt  = 0
    total_det = 0
    for idx in indices:
        gt_cnt, det_cnt = run_inference(
            model, dataset, device, idx,
            threshold = args.threshold,
            save_dir  = SAVE_DIR,
            grid_size = GRID_SIZE,
        )
        total_gt  += gt_cnt
        total_det += det_cnt

    print(f"\n  [최종 요약]")
    print(f"  총 GT 객체  : {total_gt}개")
    print(f"  총 탐지 수  : {total_det}개")
    recall_approx = min(total_det / total_gt, 1.0) if total_gt > 0 else 0
    print(f"  탐지율 (근사): {recall_approx * 100:.1f}%")
    print(f"\n  결과 저장 위치: {SAVE_DIR}")


if __name__ == '__main__':
    main()