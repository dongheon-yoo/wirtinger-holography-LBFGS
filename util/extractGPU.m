function outArray = extractGPU(inArray)
    %% Extract array from GPU to local device
    if isa(inArray, 'gpuArray')
        outArray = gather(inArray);
    else
        outArray = inArray;
    end
end
