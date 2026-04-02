"""
visualize_data.py
─────────────────────────────────────────────────────────────────
UAV 멀티모달 데이터셋 시각화 스크립트.

기능:
  1. RGB 이미지 + BBox (다중 객체 지원)
  2. Radar RD Map + BBox
  3. 8x8 그리드 기반 Confidence 히트맵 (어느 셀에 객체가 있는지 직관적으로 확인)

사용법:
  python visualize_data.py
  python visualize_data.py --file data/raw_multi/sample_0001.mat
  python visualize_data.py --dir  data/raw_crowded --n 5
─────────────────────────────────────────────────────────────────
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import koreanize_matplotlib  # 이 한 줄로 한글 폰트 자동 적용

# ── 설정 ─────────────────────────────────────────────────────────
GRID_SIZE = 8       # 그리드 크기 (8x8)
IMG_SIZE  = 64      # 이미지/레이더 해상도


def load_mat(path):
    """
    .mat 파일 로드 및 단일/다중 객체 자동 감지.
    Returns:
        img    (np.ndarray): (H, W, 3), uint8 or float
        radar  (np.ndarray): (H, W), float
        bboxes (np.ndarray): (N, 4), [cx, cy, w, h] 정규화 0~1
    """
    data = sio.loadmat(path)
    img   = data['img']
    radar = data['radar']

    if 'all_bboxes' in data:
        bboxes = np.array(data['all_bboxes'], dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)
        mode = 'multi'
    elif 'bbox' in data:
        bboxes = np.array(data['bbox'], dtype=np.float32).reshape(1, -1)
        mode = 'single'
    else:
        raise KeyError(f"파일에 'bbox' 또는 'all_bboxes' 키가 없습니다: {path}")

    print(f"  ▶ 모드: {mode} | 객체 수: {len(bboxes)}")
    return img, radar, bboxes


def build_confidence_heatmap(bboxes, grid_size=GRID_SIZE):
    """
    BBox 목록으로부터 grid_size x grid_size Confidence 히트맵 생성.
    Occlusion(동일 셀 충돌) 발생 시 해당 셀 값을 2로 표시.
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    for bbox in bboxes:
        cx, cy, w, h = bbox
        gx = min(int(cx * grid_size), grid_size - 1)
        gy = min(int(cy * grid_size), grid_size - 1)

        if heatmap[gy, gx] == 0:
            heatmap[gy, gx] = 1.0   # 정상 할당
        else:
            heatmap[gy, gx] = 2.0   # Occlusion 발생 셀 (충돌)

    return heatmap


def draw_bboxes_on_ax(ax, bboxes, img_size=IMG_SIZE, color='red', label_prefix='Obj'):
    """
    Axes에 다중 BBox를 그림.
    bbox: [cx, cy, w, h] (정규화 0~1) → 픽셀 좌표 변환 후 Rectangle 추가.
    """
    colors = list(mcolors.TABLEAU_COLORS.values())  # 객체마다 다른 색상

    for i, bbox in enumerate(bboxes):
        cx, cy, w, h = bbox
        px = (cx - w / 2) * img_size
        py = (cy - h / 2) * img_size
        pw = w * img_size
        ph = h * img_size
        c  = colors[i % len(colors)]

        ax.add_patch(patches.Rectangle(
            (px, py), pw, ph,
            linewidth=2, edgecolor=c, facecolor='none'
        ))
        ax.text(px, py - 3, f'{label_prefix}{i+1}',
                color=c, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.4, pad=1))


def visualize_single(mat_path, save_dir=None, grid_size=GRID_SIZE):
    """
    단일 .mat 파일 시각화: RGB | Radar | Confidence Heatmap 3-panel.
    """
    fname = os.path.basename(mat_path)
    print(f"\n[시각화] {fname}")

    img, radar, bboxes = load_mat(mat_path)
    heatmap = build_confidence_heatmap(bboxes, grid_size)

    # 레이더 정규화 (시각화용)
    r_min, r_max = radar.min(), radar.max()
    radar_norm = (radar - r_min) / (r_max - r_min + 1e-8)

    # Occlusion 체크
    n_occluded = int((heatmap == 2.0).sum())
    if n_occluded > 0:
        print(f"  ⚠️  Occlusion 감지: {n_occluded}개 셀에서 충돌 발생")

    # ── Figure 구성 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{fname}  |  객체 수: {len(bboxes)}  |  그리드: {grid_size}×{grid_size}",
        fontsize=13, fontweight='bold'
    )

    # Panel 1: RGB Image + BBox
    ax0 = axes[0]
    ax0.imshow(img if img.dtype == np.uint8 else (img * 255).astype(np.uint8))
    ax0.set_title('RGB Image', fontsize=11)
    draw_bboxes_on_ax(ax0, bboxes, img_size=IMG_SIZE)
    ax0.axis('off')

    # Panel 2: Radar RD Map + BBox
    ax1 = axes[1]
    im1 = ax1.imshow(radar_norm, cmap='jet', vmin=0, vmax=1)
    ax1.set_title('Radar RD Map', fontsize=11)
    draw_bboxes_on_ax(ax1, bboxes, img_size=IMG_SIZE, color='white')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis('off')

    # Panel 3: Confidence Heatmap (그리드 오버레이)
    ax2 = axes[2]
    # 커스텀 컬러맵: 0=검정, 1=초록(정상), 2=빨강(Occlusion)
    cmap_conf = mcolors.ListedColormap(['#1a1a1a', '#2ecc71', '#e74c3c'])
    bounds    = [-0.5, 0.5, 1.5, 2.5]
    norm_conf = mcolors.BoundaryNorm(bounds, cmap_conf.N)

    im2 = ax2.imshow(heatmap, cmap=cmap_conf, norm=norm_conf,
                     interpolation='nearest', aspect='auto')
    ax2.set_title(f'Confidence Heatmap ({grid_size}×{grid_size} Grid)', fontsize=11)

    # 그리드 라인
    for x in range(grid_size + 1):
        ax2.axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax2.axhline(x - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    # 셀 인덱스 텍스트
    for gy in range(grid_size):
        for gx in range(grid_size):
            v = heatmap[gy, gx]
            if v > 0:
                label = 'HIT' if v == 1.0 else 'OCC'
                ax2.text(gx, gy, label, ha='center', va='center',
                         fontsize=7, color='white', fontweight='bold')

    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))
    ax2.set_xticklabels(range(grid_size), fontsize=7)
    ax2.set_yticklabels(range(grid_size), fontsize=7)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a1a1a', label='Empty'),
        Patch(facecolor='#2ecc71', label='Object (HIT)'),
        Patch(facecolor='#e74c3c', label='Occlusion (OCC)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               fontsize=7, framealpha=0.8)

    plt.tight_layout()

    # ── 저장 ────────────────────────────────────────────────────
    if save_dir is None:
        save_dir = os.path.dirname(mat_path) or '.'
    os.makedirs(save_dir, exist_ok=True)

    save_name = fname.replace('.mat', '_viz.png')
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 저장 완료: {save_path}")
    return save_path


def visualize_batch(data_dir, n=5, save_dir=None, grid_size=GRID_SIZE):
    """
    폴더 내 n개 파일을 순서대로 시각화.
    """
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    if len(mat_files) == 0:
        print(f"[ERROR] {data_dir} 에 .mat 파일이 없습니다.")
        return

    print(f"\n[배치 시각화] '{data_dir}' 에서 {min(n, len(mat_files))}개 처리 시작")
    for fname in mat_files[:n]:
        fpath = os.path.join(data_dir, fname)
        visualize_single(fpath, save_dir=save_dir, grid_size=grid_size)

    print(f"\n✅ 배치 시각화 완료!")


# ── Entry Point ───────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RadarVision 데이터 시각화')
    parser.add_argument('--file', type=str, default=None,
                        help='단일 .mat 파일 경로')
    parser.add_argument('--dir',  type=str, default=None,
                        help='폴더 경로 (배치 시각화)')
    parser.add_argument('--n',    type=int, default=5,
                        help='배치 시각화 시 처리할 파일 수 (기본값: 5)')
    parser.add_argument('--save', type=str, default='outputs/viz',
                        help='저장 폴더 (기본값: outputs/viz)')
    parser.add_argument('--grid', type=int, default=GRID_SIZE,
                        help=f'그리드 크기 (기본값: {GRID_SIZE})')
    args = parser.parse_args()

    if args.file:
        visualize_single(args.file, save_dir=args.save, grid_size=args.grid)
    elif args.dir:
        visualize_batch(args.dir, n=args.n, save_dir=args.save, grid_size=args.grid)
    else:
        # 기본 동작: 프로젝트 루트(RadarandVision/) 기준 실제 폴더 구조 탐색
        # 실행 위치가 src/utils/ 일 수도 있으므로 루트를 자동으로 찾아 올라감
        script_dir = os.path.dirname(os.path.abspath(__file__))  # src/utils/
        root_dir   = os.path.abspath(os.path.join(script_dir, '..', '..'))  # RadarandVision/
        candidates = [
            os.path.join(root_dir, 'data', 'raw_multi'),    # 다중 객체
            os.path.join(root_dir, 'data', 'raw_crowded'),  # 밀집 객체
            os.path.join(root_dir, 'data', 'raw'),          # 단일 객체
        ]
        for cand in candidates:
            if os.path.isdir(cand):
                files = sorted([f for f in os.listdir(cand) if f.endswith('.mat')])
                if files:
                    visualize_single(
                        os.path.join(cand, files[0]),
                        save_dir=os.path.join(root_dir, 'experiments', 'viz'),
                        grid_size=args.grid
                    )
                    break
        else:
            print("❌ 데이터 폴더를 찾을 수 없습니다. --file 또는 --dir 옵션을 사용하세요.")
            print("   예시: python visualize_data.py --dir data/raw_multi --n 3")