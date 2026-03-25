import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os

class RadarVisionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. 파일 로드
        mat_path = os.path.join(self.data_dir, self.files[idx])
        data = loadmat(mat_path)
        
        # 2. 이미지 전처리: [H, W, C] -> [C, H, W] 및 0~1 정규화
        image = torch.from_numpy(data['img']).permute(2, 0, 1).float()
        
        # 3. 레이더 전처리: [H, W] -> [1, H, W] (채널 차원 추가)
        radar = torch.from_numpy(data['radar']).unsqueeze(0).float()
        
        # 4. 정답(BBox): [x_c, y_c, w, h]
        label = torch.from_numpy(data['bbox']).float()
        
        return image, radar, label

# 사용 예시
# dataset = RadarVisionDataset('./synthetic_data/')
# loader = DataLoader(dataset, batch_size=16, shuffle=True)