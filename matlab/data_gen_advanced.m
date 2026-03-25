% data_gen_advanced.m (가장 완벽한 절대경로 버전)
clear; clc;

% 현재 이 스크립트(.m)가 있는 폴더 위치를 정확히 추적합니다.
script_dir = fileparts(mfilename('fullpath')); 

% matlab 폴더에서 한 칸 올라가서(..) data -> raw 폴더로 지정합니다.
save_path = fullfile(script_dir, '..', 'data', 'raw');

if ~exist(save_path, 'dir')
    mkdir(save_path);
end

fprintf('🚀 데이터 생성 시작! 저장 위치: %s\n', save_path);

for i = 1:500
    obj_size = 0.1 + rand() * 0.15;
    pos_x = rand() * (1 - obj_size); pos_y = rand() * (1 - obj_size);
    
    img = zeros(64, 64, 3);
    r_idx = round(pos_y*64 + 1) : round((pos_y+obj_size)*64);
    c_idx = round(pos_x*64 + 1) : round((pos_x+obj_size)*64);
    img(r_idx, c_idx, :) = 0.8 + rand(length(r_idx), length(c_idx), 3)*0.2;
    
    [X, Y] = meshgrid(1:64, 1:64);
    cx = (pos_x + obj_size/2) * 64; cy = (pos_y + obj_size/2) * 64;
    radar = exp(-((X-cx).^2 + (Y-cy).^2) / (2 * 4^2)) + randn(64, 64) * 0.15;
    
    bbox = [pos_x + obj_size/2, pos_y + obj_size/2, obj_size, obj_size];
    
    filename = fullfile(save_path, sprintf('sample_%04d.mat', i));
    save(filename, 'img', 'radar', 'bbox');
end

fprintf('✅ 데이터 500개 생성 완료!\n');