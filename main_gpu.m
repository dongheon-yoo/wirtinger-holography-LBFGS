%% Wirtinger holography using L-BFGS optimization
close all;
addpath ./fminlbfgs_version2c ./util
FT = @(x) fftshift(fft2(ifftshift(x)));
IFT = @(x) fftshift(ifft2(ifftshift(x)));
if canUseGPU
    gpuIdx = 1;
    gpuDevice(gpuIdx);
end

% Suppress warning
warning('off', 'all');

% Basic global parameters
mm = 1e-3; um = 1e-6; nm = 1e-9;
slmNh = 512; slmNw = 512;
imNmax = 512;
lambda = 520 * nm;
k = 2 * pi / lambda;
pp = 6.4 * um;
propDist = 20 * mm;
init_method = 'DP'; % 'DPwithRand', 'DP' (Double phase) or 'RP' (Random phase)
prepad = true;
im_name = 'car.png';

% Create directory
dirname = sprintf('./WH_car');
mkdir(dirname);
dirname2 = sprintf('./lambda_%03d_pp_%0.1f_prop_%03d_init_%s_prepad_%d', ...
                   floor(lambda * 1e9), ...
                   pp * 1e6, ...
                   propDist * 1e3, ...
                   init_method, ...
                   double(prepad));
mkdir([dirname dirname2]);

% Optimization
params.slmNh = slmNh;
params.slmNw = slmNw;
params.imNmax = imNmax;
params.lambda = lambda;
params.dx = pp;
params.dy = pp;
params.propDist = propDist;
params.steps_per_plot = 50;
params.dirname = [dirname dirname2];
params.steps_per_plot = 20;
params.prepad = prepad;

% Read image
im = imread_with_preprocess(im_name, params);
local_im = im;
params.local_im = local_im;
im = convertGPU(im);

figure(1);
imshow(im, []);
colormap gray;

% Initial phase
switch init_method
    case 'DPwithRand'
        im_half = imresize(local_im, [slmNh, slmNw / 2]);
        maxPhase = pi;
        im_half = sqrt(im_half) .* exp(1.j * rand(size(im_half)) * maxPhase);
        field_cplx = ASM(im_half, -z_prop, 2 * pp, pp, lambda);
        hlg = encode_double_phase(field_cplx);
        phi0 = angle(hlg);
        
    case 'DP'
        im_half = imresize(local_im, [slmNh, slmNw / 2]);
        field_cplx = ASM(sqrt(im_half), -z_prop, 2 * pp, pp, lambda);
        hlg = encode_double_phase(field_cplx);
        phi0 = angle(hlg);

    case 'RP'
        phi0 = rand(size(local_im)) .* 2 * pi;
    otherwise
end
phi_vec = phi0(:);
phi_vec = convertGPU(phi_vec);

% Derivative check (Please check only once before optimization)
[user_grad, num_grad, diff] = derivativeCheck_gpu(phi_vec, im, params);
fprintf('Difference between user gradient & numerical gradient: %e\n', diff);

%% Optimization via L-BFGS
% Basic parameters
options = optimset('fminunc');
options.Display = 'iter';
options.HessUpdate = 'bfgs';        
options.GoalsExactAchieve = 0;
options.GradObj = 'on';
options.TolFun = 1e-10;
options.FunValCheck = 'on';
options.DerivativeCheck = 'off';
options.MaxIter = 2000;
options.MaxFunEvals = 1e7;
options.TolFun = 1e-10;
options.TolX = 1e-10;

% Run optimization
% [optimPhase, history] = runopt_wgrad_lbfgs_gpu(phi_vec, im, params, options)

