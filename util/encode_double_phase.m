function hlg = encode_double_phase(field_cplx)

%% Encode complex field into pure phase via double phase
amp = abs(field_cplx) ./ max(max(field_cplx));
phase = angle(field_cplx);

pa = phase - acos(amp);
pb = phase + acos(amp);
phi = zeros(size(field_cplx, 1), 2 * size(field_cplx, 2));
phi(1 : 2 : end, 1 : 2 : end) = pa(1 : 2 : end, :);
phi(1 : 2 : end, 2 : 2 : end) = pb(1 : 2 : end, :);
phi(2 : 2 : end, 1 : 2 : end) = pb(2 : 2 : end, :);
phi(2 : 2 : end, 2 : 2 : end) = pa(2 : 2 : end, :);
hlg = exp(1.j * phi);

end
