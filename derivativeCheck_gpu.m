function [user_grad, num_grad, diff] = derivativeCheck_gpu(initVariables, params)
%% Derivative check for variables
[~, ugrads] = loss_and_gradients_2D_optim(initVariables, params);

% Perturbation
mu = 1e-6;
n_delta_pert = 10;
user_grad = convertGPU(zeros(6 + n_delta_pert, 1)); user_grad(1 : 6) = ugrads(1 : 6); 
num_grad = convertGPU(zeros(6 + n_delta_pert, 1)); 
for i = 1 : 6
    varCopy = initVariables;
    var = varCopy(i);
    var = var + mu;
    varCopy(i) = var;
    [loss1, ~] = loss_and_gradients_2D_optim(varCopy, params);
    
    varCopy = initVariables;
    var = varCopy(i);
    var = var - mu;
    varCopy(i) = var;
    [loss2, ~] = loss_and_gradients_2D_optim(varCopy, params);
    
    ngrad = (loss1 - loss2) ./ (2 * mu);
    num_grad(i) = ngrad;
end

n_delta = length(initVariables(7 : end));
for i = 1 : n_delta_pert
    idx = randi(n_delta);
    varCopy = initVariables;
    var = varCopy(6 + idx);
    var = var + mu;
    varCopy(6 + idx) = var;
    [loss1, ~] = loss_and_gradients_2D_optim(varCopy, params);
    
    varCopy = initVariables;
    var = varCopy(6 + idx);
    var = var - mu;
    varCopy(6 + idx) = var;
    [loss2, ~] = loss_and_gradients_2D_optim(varCopy, params);
    ngrad = (loss1 - loss2) ./ (2 * mu);
    num_grad(6 + i) = ngrad;
    user_grad(6 + i) = ugrads(6 + idx);
end

numerator = sqrt(sum((num_grad - user_grad) .^ 2, 'all'));
denominator = sqrt(sum(num_grad .^ 2, 'all')) + sqrt(sum(user_grad .^ 2, 'all'));
diff = numerator ./ denominator;

end
