% matlab/data_gen_multi.m (새 파일로 만드셔도 좋습니다)
clear; clc; close all;

num_samples = 500;
img_size = 64;
num_objects = 2; % 물체 개수를 2개로 고정 (난이도 업!)
% 1. 현재 실행 중인 스크립트의 경로를 찾습니다.
current_script_path = fileparts(mfilename('fullpath'));

% 2. 스크립트가 'project_root/matlab' 폴더에 있다면, 한 단계 위가 루트입니다.
% 만약 스크립트가 루트에 있다면 cd(current_script_path)만 하면 됩니다.
project_root = fullfile(current_script_path, '..'); 

% 3. 최종 저장 경로 설정 (project_root/data/raw_multi)
save_dir = fullfile(project_root, 'data', 'raw_multi');

% 폴더가 없으면 생성
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

fprintf('📂 데이터 저장 경로: %s\n', save_dir);

disp('👯 [다중 객체] 데이터 생성 시작...');

for i = 1:num_samples
    img = randn(img_size, img_size, 3) * 0.1 + 0.3; % 기본 배경
    radar = randn(img_size, img_size) * 0.05;
    [X, Y] = meshgrid(1:img_size, 1:img_size);
    
    all_bboxes = zeros(num_objects, 4);

    for obj = 1:num_objects
        % 1. 객체 무작위 생성
        w = randi([8, 15]); h = randi([8, 15]);
        x = randi([1, img_size - w]); y = randi([1, img_size - h]);
        
        % 2. Vision (이미지에 물체 그리기)
        obj_color = rand(1, 1, 3) * 0.5 + 0.3;
        img(y:y+h-1, x:x+w-1, :) = img(y:y+h-1, x:x+w-1, :) + obj_color;
        
        % 3. Radar (히트맵 추가)
        cx = x + w/2; cy = y + h/2;
        sigma = (w + h) / 6;
        radar_blob = exp(-((X - cx).^2 + (Y - cy).^2) / (2 * sigma^2));
        radar = radar + radar_blob;
        
        % 4. BBox 저장 (정규화)
        all_bboxes(obj, :) = [cx/img_size, cy/img_size, w/img_size, h/img_size];
    end
    
    img = min(max(img, 0), 1);
    radar = min(max(radar, 0), 1);

    % 저장 (bboxes는 이제 2x4 행렬입니다)
    filename = fullfile(save_dir, sprintf('sample_%04d.mat', i));
    save(filename, 'img', 'radar', 'all_bboxes');
end

disp('✅ 다중 객체 데이터 500개 생성 완료!');