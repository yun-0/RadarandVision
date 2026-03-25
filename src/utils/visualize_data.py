import os
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("🚀 [Step 1] 파이썬 스크립트 실행 시작!")

# 현재 터미널이 열려있는 위치(프로젝트 폴더)를 기준으로 절대 경로 강제 생성
root_dir = os.getcwd()
mat_file = os.path.join(root_dir, 'data', 'raw', 'sample_0001.mat')

print(f"🔍 [Step 2] 데이터 찾는 중... (경로: {mat_file})")

if not os.path.exists(mat_file):
    print("❌ [실패] 지정된 경로에 파일이 없습니다! 터미널 위치가 RadarandVision 폴더인지 확인해주세요.")
else:
    print("✅ [성공] 데이터를 찾았습니다! 이미지 그리기를 시작합니다.")
    
    # 1. 데이터 로드
    data = sio.loadmat(mat_file)
    img = data['img']
    radar = data['radar']
    bbox = data['bbox'][0]

    # 2. 좌표 변환
    cx, cy, w, h = bbox
    px, py, pw, ph = (cx - w/2) * 64, (cy - h/2) * 64, w * 64, h * 64

    # 3. 그림 그리기
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('RGB Image')
    axes[0].add_patch(patches.Rectangle((px, py), pw, ph, linewidth=2, edgecolor='r', facecolor='none'))
    
    im = axes[1].imshow(radar, cmap='jet')
    axes[1].set_title('Radar RD Map')
    axes[1].add_patch(patches.Rectangle((px, py), pw, ph, linewidth=2, edgecolor='w', facecolor='none', linestyle='--'))
    fig.colorbar(im, ax=axes[1])

    # 4. 저장 (프로젝트 최상위 폴더에 저장)
    save_img_path = os.path.join(root_dir, '대망의_확인용_이미지.png')
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    
    print(f"📸 [Step 3] 찰칵! 이미지 저장 완료: {save_img_path}")