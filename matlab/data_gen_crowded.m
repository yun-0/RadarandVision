% matlab/data_gen_crowded.m
clear; clc; close all;

num_samples = 1000; % 데이터 양을 조금 늘려봅시다 (1000개)
img_size = 64;
save_dir = './data/raw_crowded/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

fprintf('👯 [Crowded Scene] 데이터 %d개 생성 시작...\n', num_samples);

for i = 1:num_samples
    img = randn(img_size, img_size, 3) * 0.1 + 0.3; % 배경
    radar = randn(img_size, img_size) * 0.05; % 레이더 노이즈
    [X, Y] = meshgrid(1:img_size, 1:img_size);
    
    % [변경] 물체 개수를 1~5개 사이로 랜덤하게 설정
    num_objects = randi([1, 5]); 
    all_bboxes = zeros(num_objects, 4);

    for obj = 1:num_objects
        w = randi([6, 12]); h = randi([6, 12]); % 크기도 살짝 다양하게
        x = randi([1, img_size - w]); y = randi([1, img_size - h]);
        
        % Vision (안개 속 흐릿한 물체)
        obj_color = rand(1, 1, 3) * 0.5 + 0.3;
        img(y:y+h-1, x:x+w-1, :) = img(y:y+h-1, x:x+w-1, :) + obj_color;
        
        % Radar (물체 위치에 히트맵)
        cx = x + w/2; cy = y + h/2;
        sigma = (w + h) / 7;
        radar_blob = exp(-((X - cx).^2 + (Y - cy).^2) / (2 * sigma^2));
        radar = radar + radar_blob;
        
        all_bboxes(obj, :) = [cx/img_size, cy/img_size, w/img_size, h/img_size];
    end
    
    img = min(max(img, 0), 1);
    radar = min(max(radar, 0), 1);

    filename = fullfile(save_dir, sprintf('sample_%04d.mat', i));
    save(filename, 'img', 'radar', 'all_bboxes');
end

disp('✅ 복합 다중 객체 데이터 생성 완료!');