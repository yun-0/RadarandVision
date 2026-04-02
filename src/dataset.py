import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np


class RadarVisionDataset(Dataset):
    """
    mmWave Radar(RD Map) + RGB Image 융합 데이터셋.
    - 단일 객체 데이터 (bbox)와 다중 객체 데이터 (all_bboxes) 모두 지원.
    - 8x8 그리드 기반 다중 객체 탐지 라벨 생성.
    - Occlusion(동일 셀 충돌) 발생 시 로그 기록.
    """

    def __init__(self, data_dir, grid_size=8, img_size=64):
        """
        Args:
            data_dir (str): .mat 파일이 있는 폴더 경로
                            (raw_multi 또는 raw_crowded 또는 단일 객체 폴더)
            grid_size (int): 그리드 분할 수 (기본값 8 → 8x8=64칸)
            img_size  (int): 이미지/레이더 해상도 (기본값 64)
        """
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.img_size = img_size
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])

        # Occlusion 통계 추적 (학습 루프에서 확인 가능)
        self.occlusion_count = 0
        self.total_objects = 0

        assert len(self.files) > 0, f"[ERROR] {data_dir} 에 .mat 파일이 없습니다."
        print(f"[Dataset] '{data_dir}' 에서 {len(self.files)}개 샘플 로드 완료 (grid={grid_size}x{grid_size})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = scipy.io.loadmat(path)

        # ── 1. 이미지 / 레이더 로드 ──────────────────────────────────────────
        # img  : (H, W, 3) → (3, H, W), float32, [0, 1] 정규화
        # radar: (H, W)    → (1, H, W), float32
        img = torch.from_numpy(data['img'].astype(np.float32)).permute(2, 0, 1)
        img = img / 255.0 if img.max() > 1.0 else img  # 0~255이면 정규화

        radar = torch.from_numpy(data['radar'].astype(np.float32)).unsqueeze(0)
        # 레이더 Min-Max 정규화 (RD Map 값 범위가 모델마다 다름)
        r_min, r_max = radar.min(), radar.max()
        if r_max > r_min:
            radar = (radar - r_min) / (r_max - r_min)

        # ── 2. BBox 로드 (단일 / 다중 자동 감지) ────────────────────────────
        bboxes = self._load_bboxes(data, path)

        # ── 3. 그리드 라벨 생성 [grid_size, grid_size, 5] ───────────────────
        # 채널: [Confidence, cx, cy, w, h]  (모두 정규화된 0~1 값)
        label = self._build_label(bboxes)

        return img, radar, label

    # ── Private Methods ───────────────────────────────────────────────────────

    def _load_bboxes(self, data, path):
        """
        단일(bbox) / 다중(all_bboxes) 키를 자동 감지하여 numpy 배열로 반환.
        Returns:
            bboxes (np.ndarray): shape (N, 4), 각 행은 [cx, cy, w, h] (정규화 0~1)
        """
        if 'all_bboxes' in data:
            # 다중 객체: shape (N, 4)
            bboxes = np.array(data['all_bboxes'], dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes.reshape(1, -1)  # 혹시 1개짜리일 경우 대비

        elif 'bbox' in data:
            # 단일 객체: shape (4,) 또는 (1, 4)
            bboxes = np.array(data['bbox'], dtype=np.float32).reshape(1, -1)

        else:
            raise KeyError(
                f"[ERROR] {os.path.basename(path)} 에 'bbox' 또는 'all_bboxes' 키가 없습니다.\n"
                f"  존재하는 키: {[k for k in data.keys() if not k.startswith('_')]}"
            )

        return bboxes

    def _build_label(self, bboxes):
        """
        객체 목록을 grid_size x grid_size 그리드 라벨로 변환.
        Occlusion(동일 셀 충돌) 발생 시 기존 객체 유지 + 통계 기록.

        Args:
            bboxes (np.ndarray): shape (N, 4), [cx, cy, w, h]
        Returns:
            label (torch.Tensor): shape (grid_size, grid_size, 5)
        """
        G = self.grid_size
        label = torch.zeros((G, G, 5), dtype=torch.float32)

        for bbox in bboxes:
            cx, cy, w, h = bbox
            self.total_objects += 1

            # 그리드 셀 인덱스 계산
            grid_x = int(cx * G)
            grid_y = int(cy * G)
            grid_x = min(max(grid_x, 0), G - 1)  # 범위 클램핑
            grid_y = min(max(grid_y, 0), G - 1)

            if label[grid_y, grid_x, 0] == 0:
                # 빈 셀: 정상 할당 (numpy.float32 → float 변환)
                label[grid_y, grid_x, 0] = 1.0
                label[grid_y, grid_x, 1] = float(cx)
                label[grid_y, grid_x, 2] = float(cy)
                label[grid_y, grid_x, 3] = float(w)
                label[grid_y, grid_x, 4] = float(h)
            else:
                # 셀 충돌 (Occlusion): 기존 객체 유지, 통계만 기록
                self.occlusion_count += 1

        return label

    def get_occlusion_stats(self):
        """학습 전체에 걸친 Occlusion 통계 반환."""
        if self.total_objects == 0:
            return {"total": 0, "occluded": 0, "rate": 0.0}
        return {
            "total": self.total_objects,
            "occluded": self.occlusion_count,
            "rate": self.occlusion_count / self.total_objects
        }