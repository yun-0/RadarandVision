"""
src/metrics.py — mAP (Mean Average Precision) 평가 모듈

IEEE 논문 표준 지표:
  - mAP@0.5   : PASCAL VOC 기준 (IoU ≥ 0.5)
  - mAP@0.5:0.95 : COCO 기준 (IoU 0.5 ~ 0.95, step 0.05 평균)

외부 의존성 없음 (torch, numpy만 사용) — Jetson NX 배포 호환성 보장
"""

import numpy as np
import torch


def decode_grid_to_boxes(pred_grid, conf_threshold=0.01):
    """
    (G, G, 5) 예측 그리드를 감지 박스 리스트로 변환.

    낮은 threshold(0.01) 사용 이유:
        PR 곡선 계산 시 모든 감지 후보를 수집해야 하므로
        Recall 100% 지점까지의 precision 변화를 포착해야 함.

    Args:
        pred_grid      : torch.Tensor or numpy (G, G, 5) — [conf, cx, cy, w, h]
        conf_threshold : float — 이 값 이상인 셀만 감지로 간주

    Returns:
        list of [conf, cx, cy, w, h] — confidence 내림차순 정렬됨
    """
    if isinstance(pred_grid, torch.Tensor):
        pred_grid = pred_grid.cpu().numpy()

    G = pred_grid.shape[0]
    boxes = []
    for gy in range(G):
        for gx in range(G):
            conf = float(pred_grid[gy, gx, 0])
            if conf > conf_threshold:
                cx = float(pred_grid[gy, gx, 1])
                cy = float(pred_grid[gy, gx, 2])
                w  = float(pred_grid[gy, gx, 3])
                h  = float(pred_grid[gy, gx, 4])
                boxes.append([conf, cx, cy, w, h])

    boxes.sort(key=lambda x: -x[0])  # confidence 내림차순
    return boxes


def compute_iou(box1, box2):
    """
    두 박스의 IoU 계산.

    Args:
        box1, box2: [cx, cy, w, h] normalized [0, 1]

    Returns:
        float — IoU 값 [0, 1]
    """
    x1_min = box1[0] - box1[2] / 2; x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2; y1_max = box1[1] + box1[3] / 2
    x2_min = box2[0] - box2[2] / 2; x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2; y2_max = box2[1] + box2[3] / 2

    inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter   = inter_w * inter_h
    area1   = box1[2] * box1[3]
    area2   = box2[2] * box2[3]
    union   = area1 + area2 - inter
    return inter / (union + 1e-6)


def compute_map(all_preds, all_gts, iou_threshold=0.5):
    """
    단일 IoU threshold에서 Average Precision 계산 (VOC 11-point 보간).

    Args:
        all_preds : list of N lists — 각 원소 = [[conf, cx, cy, w, h], ...]
        all_gts   : list of N lists — 각 원소 = [[cx, cy, w, h], ...]
        iou_threshold : float

    Returns:
        ap (float): Average Precision
    """
    # 전체 감지를 (conf, img_idx, cx, cy, w, h) 형태로 평탄화
    detections = []
    for img_idx, preds in enumerate(all_preds):
        for p in preds:
            detections.append((p[0], img_idx, p[1], p[2], p[3], p[4]))

    # confidence 내림차순 정렬
    detections.sort(key=lambda x: -x[0])

    total_gt = sum(len(g) for g in all_gts)
    if total_gt == 0:
        return 0.0

    # GT 매칭 여부 추적
    gt_matched = [np.zeros(len(g), dtype=bool) for g in all_gts]

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for i, (conf, img_idx, cx, cy, w, h) in enumerate(detections):
        det_box  = [cx, cy, w, h]
        gts_here = all_gts[img_idx]
        best_iou = 0.0
        best_j   = -1

        for j, gt_box in enumerate(gts_here):
            iou_val = compute_iou(det_box, gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_j   = j

        if best_iou >= iou_threshold and best_j >= 0:
            if not gt_matched[img_idx][best_j]:
                tp[i] = 1
                gt_matched[img_idx][best_j] = True
            else:
                fp[i] = 1  # 이미 매칭된 GT에 중복 감지
        else:
            fp[i] = 1  # IoU 미달

    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(fp)
    recall    = cum_tp / (total_gt + 1e-6)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    # VOC 11-point 보간 (recall 0.0 ~ 1.0, step 0.1)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11.0

    return float(ap)


def compute_map_range(all_preds, all_gts, iou_start=0.5, iou_end=0.95, iou_step=0.05):
    """
    COCO 스타일 mAP@[iou_start:iou_step:iou_end] 계산.

    Args:
        all_preds  : list of N lists (decode_grid_to_boxes 출력)
        all_gts    : list of N lists ([[cx, cy, w, h], ...])
        iou_start  : float (기본 0.5)
        iou_end    : float (기본 0.95)
        iou_step   : float (기본 0.05)

    Returns:
        float — IoU threshold 범위 평균 mAP
    """
    thresholds = np.arange(iou_start, iou_end + iou_step / 2, iou_step)
    aps = [compute_map(all_preds, all_gts, float(t)) for t in thresholds]
    return float(np.mean(aps))
