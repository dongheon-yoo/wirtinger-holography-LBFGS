function IProp = reconFromPhase(phase_vec, params)
%% Reconstruction from phase
propDist = params.propDist;
dx = params.dx; dy = params.dy; 
slmNh = params.slmNh; slmNw = params.slmNw;
lambda = params.lambda;

phase = reshape(phase_vec, [slmNh, slmNw]);

% Creat complex field
field = exp(1.j .* phase);

% Reconstruction via ASM
fieldProp = ASM(field, propDist, dx, dy, lambda);
IProp = abs(fieldProp) .^ 2;
% Global scaling
IProp = max(min(IProp ./ max(IProp(:)), 1), 0);

end
