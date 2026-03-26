% matlab/data_gen_advanced.m
clear; clc; close all;

num_samples = 500;
img_size = 64;
save_dir = '../data/raw/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

disp('🌪️ [극한 환경] 데이터 생성 시작 (안개 낀 카메라 & 든든한 레이더)...');

for i = 1:num_samples
    % 1. 객체 크기 및 위치 (무작위)
    w = randi([10, 20]);
    h = randi([10, 20]);
    x = randi([1, img_size - w]);
    y = randi([1, img_size - h]);

    % ====================================================
    % 📸 [시련 1] Vision (Camera) - 짙은 안개와 흐릿한 객체
    % ====================================================
    % 배경에 강한 가우시안 노이즈(안개/눈보라)를 쫙 뿌립니다.
    img = randn(img_size, img_size, 3) * 0.2 + 0.3; 
    
    % 객체 색상을 배경과 비슷하게 아주 탁하고 흐릿하게 만듭니다.
    obj_color = rand(1, 1, 3) * 0.2 + 0.4; 
    for c = 1:3
        % 객체 위에도 노이즈를 덮어서 형태를 망가뜨립니다.
        img(y:y+h-1, x:x+w-1, c) = obj_color(c) + randn(h, w) * 0.05;
    end
    img = min(max(img, 0), 1); % 안전장치: 색상값을 0~1 사이로 고정

    % ====================================================
    % 📡 [구원자] Radar (Heatmap) - 안개를 뚫는 전파
    % ====================================================
    radar = randn(img_size, img_size) * 0.1; % 기본 약한 배경 노이즈
    [X, Y] = meshgrid(1:img_size, 1:img_size);
    cx = x + w/2;
    cy = y + h/2;
    sigma = (w + h) / 4;
    
    % 레이더 반사파 (가우시안 덩어리는 여전히 선명하게 유지!)
    radar_blob = exp(-((X - cx).^2 + (Y - cy).^2) / (2 * sigma^2));
    radar = radar + radar_blob;
    radar = min(max(radar, 0), 1);

    % 3. BBox 정답 (0~1 정규화 좌표)
    bbox = [cx/img_size, cy/img_size, w/img_size, h/img_size];

    % 저장
    filename = fullfile(save_dir, sprintf('sample_%04d.mat', i));
    save(filename, 'img', 'radar', 'bbox');
end

disp('✅ 가혹한 모의고사 데이터 500개 생성 완료! (Python으로 넘어오세요!)');