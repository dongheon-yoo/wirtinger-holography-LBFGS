function outGPU = convertGPU(input)
    %% Check GPU compatibility
    if canUseGPU
        outGPU = gpuArray(input);
    else
        outGPU = input;
    end
end
