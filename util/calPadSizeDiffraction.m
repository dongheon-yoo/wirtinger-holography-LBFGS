function padSize = calPadSizeDiffraction(params)
%% Calculate padding size considering diffraction angle
lambda = params.lambda;
pp = params.dx;
propDist = params.propDist;

theta = asin(lambda / (2 * pp));
marg = abs(propDist) * tan(theta);
padSize = ceil(marg / pp);

end