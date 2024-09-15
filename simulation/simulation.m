%% Simulation of LR-HSI and HR-MSI from a real HSI

%% Firstly, load the original HSI mat files
file_path="";
load(file_path); % Note: the name of variable in this file need to be set as `HSI`.
REF=HSI;
REF=(REF-min(REF,[],'all'))/(max(REF,[],'all')-min(REF,[],'all')); % Normalize process.

%% Secondly, pre-define the simulation parameters
Sigma=0.5;
downsample_ratio=8;
blur_kernel=fspecial('gaussian', [downsample_ratio,downsample_ratio], Sigma);
[W,H,C]=size(REF);

%% Generate LR-HSI and HR-MSI
% LR-HSI
HSI=imfilter(REF,blur_kernel,'conv');
HSI=HSI(1:downsample_ratio:W,1:downsample_ratio:H,:);

% HR-MSI
srf_path="";
load(srf_path); % Note: the name of spectral response function (SRF) need to be set as `srf`.
SRF=srf;
MSI=reshape(reshape(REF,[W*H,C])*(SRF')./sum(SRF,2)',[W,H,size(SRF,1)]);

%% Optional: Add Noise
SNRh=35;
sigmah = sqrt(sum(HSI(:).^2)/(10^(SNRh/10))/numel(HSI));
HSI=sigmah*randn(size(HSI));

SNRm=35;
sigmam = sqrt(sum(MSI(:).^2)/(10^(SNRh/10))/numel(MSI));
MSI=sigmam*randn(size(MSI));

%% Save the Results
output_path="";
save(output_path,"REF","MSI","HSI");
