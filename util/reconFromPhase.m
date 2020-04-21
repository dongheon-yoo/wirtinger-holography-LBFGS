function blurProp = reconFromVarWirtinger(variables, params)
%% Reconstruction from variables
propDist = params.propDist;
blurProp = cell(1, length(propDist));
dx = params.dx; dy = params.dy; 
slmNh = params.slmNh; slmNw = params.slmNw;
lambda = params.lambda;

optimDelta = reshape(variables, [slmNh, slmNw]);

% Creat complex field
field = exp(1.j .* optimDelta);

% Reconstruction via ASM
for i = 1 : length(propDist)
    fieldProp = ASM(field, propDist(i), dx, dy, lambda);
    IProp = abs(fieldProp) .^ 2;
    % Global scaling
    IProp = IProp ./ max(IProp(:));
    blurProp{i} = max(min(IProp, 1), 0);
end

end
