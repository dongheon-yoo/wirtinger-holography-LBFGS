function [loss, gradients] = loss_and_gradients(phase_vec, image, params)
%% Compute loss (L2)                                            
% Basic parameters
slmNh = params.slmNh; slmNw = params.slmNw;
dx = params.dx; dy = params.dy; 
lambda = params.lambda;
propDist = params.propDist;

% Unzip variables
phase = reshape(phase_vec, [slmNh, slmNw]);

% Create complex field
field = exp(1.j .* phase);

% Reconstruction via ASM
fieldProp = ASM(field, propDist, dx, dy, lambda);
IProp = abs(fieldProp) .^ 2;

% Compute L2 loss
loss = sum((IProp - image) .^ 2, 'all');

%% Compute gradients
if nargout > 1 % gradient required
    % Notation from "Wirtinger Holography"
    delta_f = 2 .* (IProp - image) .* 2 .* fieldProp;
    delta_f = ASM(delta_f, -propDist, dx, dy, lambda);
    gradients = real(-1.j .* exp(-1.j .* phase) .* delta_f);
    gradients = convertGPU(gradients(:));
end
