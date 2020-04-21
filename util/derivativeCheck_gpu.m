function [user_grad, num_grad, diff] = derivativeCheck_gpu(phase_vec, image, params)
%% Derivative check for gradient-based optimization
[~, ugrads] = loss_and_gradients(phase_vec, image, params);

% Perturbation
mu = 2e-6 * norm(phase_vec);
n_pert = 10;
user_grad = convertGPU(zeros(n_pert, 1)); 
num_grad = convertGPU(zeros(n_pert, 1)); 

for i = 1 : n_pert
    idx = randi(length(phase_vec));
    varCopy = phase_vec;
    var = varCopy(idx);
    var = var + mu;
    varCopy(idx) = var;
    [loss1, ~] = loss_and_gradients(varCopy, image, params);
    
    varCopy = phase_vec;
    var = varCopy(idx);
    var = var - mu;
    varCopy(idx) = var;
    [loss2, ~] = loss_and_gradients(varCopy, image, params);
    ngrad = (loss1 - loss2) ./ (2 * mu);
    num_grad(i) = ngrad;
    user_grad(i) = ugrads(idx);
end

numerator = sqrt(sum((num_grad - user_grad) .^ 2, 'all'));
denominator = sqrt(sum(num_grad .^ 2, 'all')) + sqrt(sum(user_grad .^ 2, 'all'));
diff = numerator ./ denominator;

end
