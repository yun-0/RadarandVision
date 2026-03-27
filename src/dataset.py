import torch
from torch.utils.data import Dataset
import scipy.io
import os
import numpy as np

class RadarVisionDataset(Dataset):
    def __init__(self, data_dir, grid_size=4):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        self.grid_size = grid_size # 4x4 그리드

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = scipy.io.loadmat(path)

        # 이미지/레이더 전처리 (기존과 동일)
        img = torch.from_numpy(data['img']).float().permute(2, 0, 1)
        radar = torch.from_numpy(data['radar']).float().unsqueeze(0)
        
        # [핵심] 다중 객체용 그리드 라벨 생성 (4, 4, 5)
        # 5개 채널: [Confidence, x, y, w, h]
        label = torch.zeros((self.grid_size, self.grid_size, 5))
        
        bboxes = data['all_bboxes'] # MATLAB에서 만든 [num_objects, 4]
        
        for bbox in bboxes:
            cx, cy, w, h = bbox
            # 어느 그리드 칸에 속하는지 계산
            grid_x = int(cx * self.grid_size)
            grid_y = int(cy * self.grid_size)
            
            # 인덱스 범위 초과 방지
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)
            
            # 해당 칸에 물체가 있다고 표시 (Confidence = 1)
            if label[grid_y, grid_x, 0] == 0: # 한 칸에 물체 하나만 할당
                label[grid_y, grid_x, 0] = 1.0
                label[grid_y, grid_x, 1:] = torch.tensor([cx, cy, w, h])

        return img, radar, label