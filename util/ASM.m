function fieldProp = ASM(field, propDist, dx, dy, wavelength, clip)
%% Written by Jaebum Cho, OEQE Laboratory, Seoul National University

% Set zero padding size
angX = asin(wavelength / (2 * dx)); angY = asin(wavelength / (2 * dy));
margX = abs(propDist) * tan(angX); margX = ceil(margX / dx);
margY = abs(propDist) * tan(angY); margY = ceil(margY / dy);
fieldPad = padarray(field, [margY,margX]);

% Set Spatial Frequency Domain
[padNy, padNx] = size(fieldPad);
Tfx = 1 / dx; Tfy = 1 / dy;
dfx = Tfx / nx; dfy = Tfy / ny;
if rem(nx, 2) == 0, fx = -Tfx / 2 : dfx : Tfx / 2 - dfx;
else fx = -(Tfx - dfx) / 2 : dfx : (Tfx - dfx) / 2; end
if rem(ny, 2) == 0, fy = -Tfy / 2 : dfy : Tfy / 2 - dfy;
else fy = -(Tfy - dfy) / 2 : dfy : (Tfy - dfy) / 2; end

[Fx, Fy] = meshgrid(fx, fy);

Gamma=(1 / wavelength)^2 - Fx.^2 - Fy.^2;
AS = fftshift(fft2(ifftshift(field)));
F = fftshift(ifft2(ifftshift(AS .* exp(2.j * pi * sqrt(Gamma) * propDist))));
if nargin > 5
    if clip
        fieldProp = F(1 + margY : end - margY, 1 + margX : end - margX);    
    end
else
    fieldProp = F(1 + margY : end - margY, 1 + margX : end - margX);    
end


