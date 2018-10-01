close all;
clear all; 
%%
run matconvnet\matlab\vl_setupnn;

addpath('utils_compress')
load('models\DACAR_keras.mat')
%load('models\SACAR_keras.mat')

% to save Results 
mkdir('Result');
%%

% for LIVE1 file
im_path = 'Data\LIVE1\';  
im_dir = dir( fullfile(im_path, '*bmp') );

% for BSD100 file
% im_path = 'Data\BSD100\';  
% im_dir = dir( fullfile(im_path, '*jpg') );

im_num = length( im_dir );

%% to save scores:
PSNR_scores(im_num,2)=0;
PSNR_B_scores(im_num,2)=0;
BEF_scores(im_num,2)=0;
MSSIM_scores(im_num,2)=0; %% used as SSIM 
SSIM_scores(im_num,2)=0;  

%% set parameters
JPEG_Quality = 10; % or = 20

for img = 1:im_num
    
    im = imread( fullfile(im_path, im_dir(img).name) );
    
    % work on illuminance only
    if size(im,3)>1
       im_ycbcr = rgb2ycbcr(im);
       im = im_ycbcr(:, :, 1);
    end
    im_gnd = im2double(im);

    %% generate JPEG-compressed input
    
    imwrite(im_gnd,'im_JPEG.jpg','jpg','Quality',JPEG_Quality);
    im_input = im2double(imread('im_JPEG.jpg'));
    
    imwrite(im_gnd, ['Result\', im_dir(img).name]);
    imwrite(im_input, ['Result\', im_dir(img).name, '_im_input.jpg'] );   
    
    %% DACAR or SACAR CNN:
    
    im_output = DACAR(im_input, model); % 3, 4, 5 layers DACAR modl in our paper (Dirct art.)
    %im_output = SACAR(im_input, model);  % 6, 7 layers SACAR modl in our paper (skip art.)
  
    % for BSD100 file
    %imwrite(im_output, ['Result\', im_dir(img).name, '_im_output.jpg'] );
    
    % for LIVE1 file
    imwrite(im_output, ['Result\', im_dir(img).name, '_im_output.bmp'] );
    
%%     % compute errors
%     error_input = compute_errors(im_gnd, im_input);
%     error_output = compute_errors(im_gnd, im_output);
    
    % Show
%     if size(im_gnd, 3) == 3
%        im_gnd = rgb2ycbcr(im_gnd);
%        im_gnd = im_gnd(:, :, 1);
%     end

%     if size(im, 3) == 3
%        im = rgb2ycbcr(im);
%        im = im(:, :, 1);
%     end

    im_gnd = double(im_gnd);
    im_input = double(im_input);
    im_output = double(im_output);

    if max(im_gnd(:)) < 2
       im_gnd = im_gnd * 255;
    end
    if max(im_input(:)) < 2
       im_input = im_input * 255;
    end
    if max(im_output(:)) < 2
       im_output = im_output * 255;
    end
    
    
    %PSNR 
    PSNR_scores(img, 1)= compute_psnr(im_gnd, im_input);
    PSNR_scores(img, 2)= compute_psnr(im_gnd, im_output);
    
    %PSNR_B 
    PSNR_B_scores(img, 1)= compute_psnrb(im_gnd, im_input);
    PSNR_B_scores(img, 2)= compute_psnrb(im_gnd, im_output);
    
    %BEF 
    BEF_scores(img, 1)= compute_bef(im_input);
    BEF_scores(img, 2)= compute_bef(im_output);
    
    %MSSIM 
    MSSIM_scores(img, 1)= ssim_index(im_gnd, im_input, [0.01 0.03], ones(8));
    MSSIM_scores(img, 2)= ssim_index(im_gnd, im_output, [0.01 0.03], ones(8));
    
    %SSIM 
    SSIM_scores(img, 1)= ssim(im_gnd, im_input);
    SSIM_scores(img, 2)= ssim(im_gnd, im_output);
    
end    

%% scores mean

PSNR_scores(im_num+1,:)= mean(PSNR_scores);
PSNR_B_scores(im_num+1,:)= mean(PSNR_B_scores);
BEF_scores(im_num+1,:)= mean(BEF_scores);
MSSIM_scores(im_num+1,:)= mean(MSSIM_scores);
SSIM_scores(im_num+1,:)= mean(SSIM_scores);

% save scores
save Result\PSNR_scores PSNR_scores;
save Result\PSNR_B_scores PSNR_B_scores;
save Result\BEF_scores BEF_scores;
save Result\MSSIM_scores MSSIM_scores;
save Result\SSIM_scores SSIM_scores;

celldisp(names);