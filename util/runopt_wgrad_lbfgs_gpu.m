%% Run image based optimization via L-BFGS method 
function [optimPhase, history] = runopt_wgrad_lbfgs_gpu(phase_vec, image, params, options)
% Basic parameters
slmNh = params.slmNh; slmNw = params.slmNw;
history.mseVal = []; history.psnrVal = []; history.iter = [];
options.OutputFcn = @lbfgs_outfun;
propDist = params.propDist;

% Start optimization
optimFunc = @(x)loss_and_gradients(x, image, params);

% For recording
f1 = figure; 
[optimPhase, ~] = fminlbfgs_gpu(optimFunc, phase_vec, options);
    
    function stop = lbfgs_outfun(x, optimValues, state)
        stop = false;
        switch state
            case 'init'
            case 'iter'
                % Record & Plot
                if rem(optimValues.iteration, params.steps_per_plot) == 0
                    phi = reshape(x, [slmNh, slmNw]);
                    phi = angle(exp(1.j .* phi));
                    phi = (phi + pi) / (2 * pi);
                    local_phi = extractGPU(phi);
                    phase_name = [params.dirname, sprintf('/phase_%dIter.png', optimValues.iteration)];
                    imwrite(uint8(local_phi * 255), phase_name);
                    
                    psnrVal = zeros(1, length(propDist));
                    IProp = reconFromPhase(x, params);
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

