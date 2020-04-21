function [loss, gradients] = loss_and_gradients(x, fieldIdeal, params)
%% Compute loss (L2)                                            
% Basic parameters
slmNh = params.slmNh; slmNw = params.slmNw;
dx = params.dx; dy = params.dy; 
lambda = params.lambda;
z_prop = params.z_prop;

% Unzip variables
alpha1 = x(1); beta1 = x(2); gamma1 = x(3);
alpha2 = x(4); beta2 = x(5); gamma2 = x(6);
delta = x(7 : end);
delta = reshape(delta, [slmNh, slmNw]);

% Create complex field (Mode 1 -> Amplitude, Mode 2 -> Phase)
[amplitude1, grad_amp1] = calAmplitude(alpha1, beta1, gamma1, delta);
[phase2, grad_phi2] = calPhase(alpha2, beta2, gamma2, delta);
field = amplitude1 .* exp(1.j .* phase2);

% Reconstruction via ASM
fieldProp = ASM(field, z_prop, dx, dy, lambda);
fieldIdealProp = ASM(fieldIdeal, z_prop, dx, dy, lambda);
IProp = abs(fieldProp) .^ 2;
im = params.local_im;

% Notation from "Wirtinger Holography"
lossWeights = params.lossWeights;
% delta_f = 2 .* (IProp - im) .* 2 .* fieldProp;
% delta_f = 2 * (fieldProp - fieldIdeal);
delta_f = 2 * (fieldProp - fieldIdealProp) * lossWeights(1) + ...
          2 .* (IProp - im) .* 2 .* fieldProp * lossWeights(2);
delta_f = ASM(delta_f, -z_prop, dx, dy, lambda);

% Compute L2 loss
loss = sum(abs(fieldProp - fieldIdealProp) .^ 2, 'all') * lossWeights(1) + ...
       sum((IProp - im) .^ 2, 'all') * lossWeights(2);

%% Compute gradients
if nargout > 1 % gradient required
    gval = calLossGradients(amplitude1, grad_amp1, phase2, grad_phi2, delta_f);
    dalpha1 = gval{1}; dalpha2 = gval{4};
    dbeta1 = gval{2}; dbeta2 = gval{5};
    dgamma1 = gval{3}; dgamma2 = gval{6};
    ddelta = gval{7};
    gradients = [dalpha1; dbeta1; dgamma1; dalpha2; dbeta2; dgamma2; ddelta(:)];
    gradients = convertGPU(gradients);
end
