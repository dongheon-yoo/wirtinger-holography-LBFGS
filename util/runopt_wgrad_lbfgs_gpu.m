%% Run image based optimization via L-BFGS method 
function [optimPhase, history] = runopt_wgrad_lbfgs_gpu(phase_vec, image, params, options)
% Basic parameters
slmNh = params.slmNh; slmNw = params.slmNw;
history.psnrVal = []; history.iter = [];
history.variables = zeros(options.MaxIter, 6); % [alpha1, beta1, gamma1, alpha2, beta2, gamma2]
options.OutputFcn = @lbfgs_outfun;
propDist = params.propDist;
blurIm = params.blurIm;

% Start optimization
optimFunc = @(x)loss_and_gradients(x, fieldIdeal, params);

% For recording
f1 = figure; 
[optimVariables, ~] = fminlbfgs_gpu(optimFunc, initVariables, options);
    
    function stop = lbfgs_outfun(x, optimValues, state)
        stop = false;
        switch state
            case 'init'
            case 'iter'
                % Record variables
                history.variables(optimValues.iteration, :) = extractGPU(x(1 : 6).');
                local_vars = extractGPU(history.variables);
                csvwrite([params.dirname, './variables.csv'], local_vars);
           
                % Record & Plot
                if rem(optimValues.iteration, params.steps_per_plot) == 0
                    blurProp = reconFromVar(x, params);
                    delta = reshape(x(7 : end), [slmNh, slmNw]);
                    delta = angle(exp(1.j .* delta));
                    delta = (delta + pi) / (2 * pi);
                    delta = extractGPU(delta);
                    folder_name = [params.dirname, sprintf('/%d Iter', optimValues.iteration)];
                    delta_name = '/phase.png';
                    mkdir(folder_name);
                    delta_name = [folder_name, delta_name];
                    imwrite(uint8(delta * 255), delta_name);
                    
                    psnrMean = zeros(1, length(propDist));
                    for idx = 1 : length(propDist)
                        IProp = extractGPU(blurProp{idx});
                        I_name = sprintf('/recon_%.2f.png', propDist(idx) * 1e3);
                        I_name = [folder_name, I_name];
                        imwrite(uint8(IProp * 255), I_name);
                        local_im = extractGPU(blurIm{idx});
                        pval = psnr(IProp, local_im);
                        psnrMean(idx) = pval;
                    end
                    history.psnrVal = [history.psnrVal; mean(psnrMean)];
                    
                    history.iter = [history.iter; optimValues.iteration];
                    figure(f1);
                    plot(history.iter, history.psnrVal);
                    xlabel('Iterations'); ylabel('PSNR (dB)');
                    title('PSNR (dB)');
                    saveas(f1,[params.dirname, './psnr.png']);
   
                end
               
            case 'done'
            otherwise
        end
    end
end

