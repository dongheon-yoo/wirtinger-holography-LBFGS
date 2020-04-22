function out = imread_with_preprocess(path_to_image, params)
    
    %% Read image & Preprocess
    imNmax = params.imNmax;
    slmNh = params.slmNh;
    slmNw = params.slmNw;
    
    im = imread(path_to_image);
    if size(im, 3) == 3, im = rgb2gray(im); end
    if imNmax < max(size(im, [1, 2]))
        scale = imNmax / max(size(im, [1, 2]));
        im = imresize(im, scale);
    end
    
    % Pad image
    [h, w] = size(im, [1, 2]);
    padH1 = ceil((slmNh - h) / 2); padH2 = slmNh - h - padH1;
    padW1 = ceil((slmNw - w) / 2); padW2 = slmNw - w - padW1;
    im = padarray(im, [padH1, padW1], 'pre');
    im = padarray(im, [padH2, padW2], 'post');
    
    % Prepad image considering diffraction angle
    if params.prepad
        padSize = calPadSizeDiffraction(params);
        oh = slmNh - 2 * padSize;
        ow = slmNw - 2 * padSize;
        im = imresize(im, [oh, ow]);
        im = padarray(im, [padSize, padSize]);
    end
    out = im2double(im);
end
