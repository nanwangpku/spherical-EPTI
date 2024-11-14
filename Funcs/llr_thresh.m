function [ alpha_thresh, s_vals ] = llr_thresh(alpha, lambda, block_dim, randshift)
%llr_thresh Locally Low Rank regularization
%   Implements Locally Low Rank regularization through singular value soft
%   thresholding. Non-overlapping patches are extracted from the image and
%   reshaped into a matrix. Random shifting is applied before/after the
%   reshaping
%
%  Inputs:
%    alpha [ny, nz, K] -- coefficient images with K coefficients, or
%    [ny,nz,nx,K], this is for 3D
%    lambda -- soft threshold parameter
%    block_dim [Wy, Wz] -- spatial dimensions of image block or [Wy, Wz,
%    Wx] for 3D
%    randshift -- true to perform random shifting
%
%  Outputs:
%     alpha_thresh [ny, nz, K] -- coefficient image after singular value
%        thresholding  or [ny,nz,nx,K]
%     s_vals [ny / Wy, nz / Wz, K] -- singular values of each block before
%        thresholding, or [ny/Wy, nz/Wz, nx/Wx]
%
%  Notes:
%     The image dimensions should be divisible by the block sizes
if nargin < 4
        randshift = true;
end

dim = numel(block_dim)

if dim == 2 % this is 2D image + coefficient num (can be 1)

    Wy = block_dim(1);
    Wz = block_dim(2);

    [ny, nz, K] = size(alpha);
    if isgpuarray(alpha)
        useGPU = true;
    else
        useGPU = false;
    end
    % reshape into patches
    % L = (ny - (Wy - 1)) * (nz - (Wz - 1)); % for sliding
    L = ny * nz / Wy / Wz; % for distinct

    if randshift
        shift_idx = [randi(Wy) randi(Wz) 0];
        alpha = circshift(alpha, shift_idx);
    end

    % alpha_LLR = single(zeros(Wy*Wz, L, K));
    for ii=1:K
        alpha_LLR(:,:,ii) = im2col(alpha(:,:,ii), [Wy,Wz], 'distinct');
    end
    alpha_LLR = gather(permute(alpha_LLR, [1 3 2]));
    % s_LLR = zeros(K, L);
    % if useGPU
    %     alpha_LLR = gpuArray(single(alpha_LLR));
    % end
    % threshold singular values
    parfor ii=1:L
        [UU, SS, VV] = svd(alpha_LLR(:,:,ii), 'econ');
        s_LLR(:,ii) = diag(SS);
        s2 = SoftThresh(s_LLR(:,ii), lambda);
        alpha_LLR(:,:,ii) = UU*diag(s2)*VV';
    end

    % reshape into image
    % alpha_thresh = zeros(size(alpha));
    parfor ii=1:K
        alpha_thresh(:,:,ii) = col2im(alpha_LLR(:,ii,:), [Wy, Wz], [ny, nz], 'distinct');
    end

    if randshift
        alpha_thresh = circshift(alpha_thresh, -1 * shift_idx);
    end
    if useGPU
        alpha_LLR = gpuArray(single(alpha_LLR));
    end
    s_vals = permute(reshape(s_LLR, [K, ny / Wy, nz / Wz]), [2, 3, 1]);

elseif dim == 3 % this is 2D image + coefficient num (can be 1)
    Wy = block_dim(1);
    Wz = block_dim(2);
    Wx = block_dim(3);    

    [ny, nz, nx, K] = size(alpha);
    if isgpuarray(alpha)
        useGPU = true;
    else
        useGPU = false;
    end
    % reshape into patches
    % L = (ny - (Wy - 1)) * (nz - (Wz - 1)); % for sliding
    L = ny * nz * nx / Wy / Wz /Wx; % for distinct

    if randshift
        shift_idx = [randi(Wy) randi(Wz) randi(Wx) 0];
        alpha = circshift(alpha, shift_idx);
    end

   
    for ii=1:K
        alpha_LLR(:,:,ii) = im2col3(alpha(:,:,:,ii), [Wy,Wz,Wx], [Wy,Wz,Wx]);  % in this function, step size = block size means 'distinct'
    end
    alpha_LLR = gather(permute(alpha_LLR, [1 3 2]));
    
    % threshold singular values
    parfor ii=1:L
        [UU, SS, VV] = svd(alpha_LLR(:,:,ii), 'econ');
        s_LLR(:,ii) = diag(SS);
        s2 = SoftThresh(s_LLR(:,ii), lambda);
        alpha_LLR(:,:,ii) = UU*diag(s2)*VV';
    end

    % reshape into image
    % alpha_thresh = zeros(size(alpha));
    parfor ii=1:K
        alpha_thresh(:,:,:,ii) = col2im3(alpha_LLR(:,ii,:), [Wy, Wz, Wx], [ny, nz, nx], [Wy,Wz,Wx]);
    end

    if randshift
        alpha_thresh = circshift(alpha_thresh, -1 * shift_idx);
    end
    if useGPU
        alpha_LLR = gpuArray(single(alpha_LLR));
    end
    s_vals = permute(reshape(s_LLR, [K, ny / Wy, nz / Wz, nx / Wx]), [2, 3, 4, 1]);
end
