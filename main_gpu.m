%% Wirtinger holography using L-BFGS optimization
close all;
addpath ./fminlbfgs_version2c ./util
FT = @(x) fftshift(fft2(ifftshift(x)));
IFT = @(x) fftshift(ifft2(ifftshift(x)));
if canUseGPU
    gpuIdx = 1;
    gpuDevice(gpuIdx);
end

% Basic global parameters
mm = 1e-3; um = 1e-6; nm = 1e-9;
slmNh = 512; slmNw = 512;
imMax = 512;
lambda = 520 * nm;
k = 2 * pi / lambda;
pp = 6.4 * um;
propDist = 20 * mm;
init_method = 'DP'; % 'DPwithRand', 'DP' (Double phase) or 'RP' (Random phase)

% Create directory
dirname = sprintf('./WH_baboon');
mkdir(dirname);
dirname2 = sprintf('./lambda_%03d_pp_%0.1f_prop_%03d_init_%s', ...
                   floor(lambda * 1e9), ...
                   pp * 1e6, ...
                   z_prop * 1e3, ...
                   init_method);
mkdir([dirname dirname2]);

% Read image
im_name = 'baboon.png';
color_im = imread(im_name);
if size(color_im, 3) == 3, color_im = rgb2gray(color_im); end
if max(size(color_im, [1, 2])) > imMax
    scale = imMax / max(size(color_im, [1, 2]));
    color_im = imresize(color_im, scale);
end
im = im2double(color_im);
[imH, imW] = size(im, [1, 2]);
padH1 = ceil((slm_Ny - imH) / 2); padH2 = slm_Ny - imH - padH1;
padW1 = ceil((slm_Nx - imW) / 2); padW2 = slm_Nx - imW - padW1;
im = padarray(im, [padH1, padW1], 'pre');
im = padarray(im, [padH2, padW2], 'post');
im = convertGPU(im); 
im_vec = im(:);
figure;
imshow(im, []);
colormap gray;

% Optimization
params.Nx = slm_Nx;
params.Ny = slm_Ny;
params.lambda = lambda;
params.dx = pp;
params.dy = pp;
params.z_prop = z_prop;
params.dirname = [dirname dirname2];
params.steps_per_plot = 20;

options = optimset('fminunc');
options.Display = 'iter';
options.HessUpdate = 'bfgs';        
options.GoalsExactAchieve = 1;
options.GradObj = 'on';
options.TolFun = 1e-7;
options.FunValCheck = 'on';
options.DerivativeCheck = 'on';
options.MaxIter = 4000;
options.MaxFunEvals = 1e7;
options.TolFun = 1e-10;

% Initial phase
switch init_method
    case 'DPwithRand'
        im_half = imresize(im, [slm_Ny, slm_Nx / 2]);
        maxPhase = pi;
        im_half = sqrt(complex(im_half)) .* exp(1.j * rand(size(im_half)) * maxPhase);
        field_cplx = ASM(im_half, -z_prop, 2 * pp, pp, lambda);
        hlg = encode_double_phase_gpu(field_cplx);
        phi0 = angle(hlg);
        
    case 'DP'
        im_half = imresize(im, [slm_Ny, slm_Nx / 2]);
        field_cplx = ASM(sqrt(complex(im_half)), -z_prop, 2 * pp, pp, lambda);
        hlg = encode_double_phase_gpu(field_cplx);
        phi0 = angle(hlg);

    case 'RP'
        phi0 = rand(size(im)) .* 2 * pi;
    otherwise
end
phi_vec = phi0(:);
phi_vec = convertGPU(phi_vec);

% Derivative check (Please check only once before optimization)
[user_grad, num_grad, diff] = derivativeCheck(phi_vec, im_vec, params);
fprintf('Difference between user gradient & numerical gradient: %e\n', diff);


% Run optimization
% if meanShift
%     [phi_optim, mse_val, psnr_val, history] = runopt_wgrad_meanShift_gpu(phi_vec, im_vec, params, options);
% else
%     [phi_optim, mse_val, psnr_val, history] = runopt_wgrad(phi_vec, im_vec, params, options);
% end

