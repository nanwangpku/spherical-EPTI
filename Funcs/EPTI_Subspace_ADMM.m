function [im_recon,alpha,history]=EPTI_Subspace_ADMM(kdata,mask_sample,smap,Phase_total,Phi,r_ops,Func,Nstep, a_init,callback)
% Func: a string indicating which regularization function to use. Options:
%       spatial TV, LLR
% Nstep: how many intermediate x to save
% a_init: initial guess
% callback: call back functions

% Tikhonov
if nargin < 7
    a_init = [];
    callback = false;
    Nstep = 1;
    Func = 'SpaTV';
elseif nargin < 8
    a_init = [];
    callback = false;
    Nstep = 1;
elseif nargin < 9
    a_init = [];
    callback = false;
elseif nargin < 10
    callback = false;
end
[nt,nx,ny,nc] = size(kdata);
kdata=permute(kdata,[2 3 4 1]);
mask_sample=permute(mask_sample,[2 3 4 1 5]);

K=size(Phi,2);

% Subspace basis projection
dims=[nx,ny,K];

ns = size(Phase_total,5); 

% T_for = @(a) temporal_forward(a, Phi, dims);
% T_adj = @(x) temporal_adjoint(x, Phi);
% % Image Phase
% % Phase_total=Phase_T.*repmat(Phase0,[1,1,nt]);
% P_for = @(x) bsxfun(@times, x, Phase_total);
% P_adj = @(y) bsxfun(@times, y, conj(Phase_total));
% % Sampling mask
% % smap=repmat(smap,[1 1 1 nt]);
% S_for = @(a) bsxfun(@times, smap, permute(a, [1, 2, 4, 3]));
% S_adj = @(as) squeeze(sum(bsxfun(@times, conj(smap), as), 3));
% % Sampling mask
% F_for = @(x) fft2c_for(x);
% F_adj = @(y) fft2c_adj(y);
% % Sampling mask
% M_for = @(y) Mask_forward(y, mask_sample);
% M_adj = @(y) Mask_adjoint(y, mask_sample);
% 
% A_for = @(a) M_for(F_for(S_for(P_for(T_for(a)))));
% A_adj = @(y) T_adj(P_adj(S_adj(F_adj(M_adj(y)))));
% 
% AHA = @(a) A_adj(A_for(a));
if ns==1
    T_for = @(a) temporal_forward(a, Phi, dims);
    T_adj = @(x) temporal_adjoint(x, Phi);
    % Image Phase
    % Phase_total=Phase_T.*repmat(Phase0,[1,1,nt]);
    P_for = @(x) bsxfun(@times, x, Phase_total);
    P_adj = @(y) bsxfun(@times, y, conj(Phase_total));
    % Sampling mask
    % smap=repmat(smap,[1 1 1 nt]);
    S_for = @(a) bsxfun(@times, smap, permute(a, [1, 2, 4, 3]));
    S_adj = @(as) squeeze(sum(bsxfun(@times, conj(smap), as), 3));
    % Sampling mask
    F_for = @(x) fft2c_for(x);
    F_adj = @(y) fft2c_adj(y);
    % Sampling mask
    M_for = @(y) Mask_forward(y, mask_sample);
    M_adj = @(y) Mask_adjoint(y, mask_sample);

    A_for = @(a) M_for(F_for(S_for(P_for(T_for(a)))));
    A_adj = @(y) T_adj(P_adj(S_adj(F_adj(M_adj(y)))));

    AHA = @(a) A_adj(A_for(a));
    AHA_lsqr = @(a,lFlag)A_adj(A_for(a));
else
    T_for = @(a) temporal_forward(a, Phi, dims);
    T_adj = @(x) temporal_adjoint(x, Phi);

    S_for = @(a) bsxfun(@times, smap, permute(a, [1, 2, 4, 3]));
    S_adj = @(as) squeeze(sum(bsxfun(@times, conj(smap), as), 3));
    % Sampling mask
    F_for = @(x) fft2c_for(x);
    F_adj = @(y) fft2c_adj(y);

    for i = 1:ns
        % Image Phase
        % Phase_total=Phase_T.*repmat(Phase0,[1,1,nt]);
        P_for = @(x) bsxfun(@times, x, Phase_total(:,:,:,:,i));
        P_adj = @(y) bsxfun(@times, y, conj(Phase_total(:,:,:,:,i)));
        % Sampling mask
        % smap=repmat(smap,[1 1 1 nt]);

        % Sampling mask
        M_for = @(y) Mask_forward_ns(y, mask_sample(:,:,:,:,i));
        M_adj = @(y) Mask_adjoint_ns(y, mask_sample(:,:,:,:,i));
        if i == 1
            A_for = @(a) M_for(F_for(S_for(P_for(T_for(a)))));
            A_adj = @(y) T_adj(P_adj(S_adj(F_adj(M_adj(y)))));
        else
            A_for = @(a) A_for(a) + M_for(F_for(S_for(P_for(T_for(a)))));
            A_adj = @(y) A_adj(y) + T_adj(P_adj(S_adj(F_adj(M_adj(y)))));
        end
    end

    AHA = @(a) A_adj(A_for(a));
    AHA_lsqr = @(a,lFlag)A_adj(A_for(a));
end

%% scaling
tmp = dimnorm(ifft2c(kdata), 3);
tmpnorm = dimnorm(tmp, 4);
tmpnorm2 = sort(tmpnorm(:), 'ascend');
% match convention used in BART
p100 = tmpnorm2(end);
p90 = tmpnorm2(round(.9 * length(tmpnorm2)));
p50 = tmpnorm2(round(.5 * length(tmpnorm2)));
if (p100 - p90) < 2 * (p90 - p50)
    scaling = p90;
else
    scaling = p100;
end

% scaling=1;
fprintf('\nScaling: %f\n\n', scaling);

kdata = kdata ./ scaling;
ksp_adj = reshape(A_adj(kdata),dims);
%% ADMM
tic;

alpha_ref = RefValue;
alpha_ref.data = a_init ./ scaling;
switch Func
    case 'LLR'
        r_ops.objfun = @(a, sv, lam) 0.5*norm_mat(kdata(:) - A_for(a))^2 + lam*sum(sv(:));
        r_ops.dc = @(a) norm_mat(kdata(:) - A_for(a))^2;
        r_ops.reg = @(sv) sum(sv(:));
        if callback
            history = iter_llr(alpha_ref, r_ops,  AHA, ksp_adj, Nstep, @callback_admm);
        else
            history = iter_llr(alpha_ref,  r_ops,  AHA, ksp_adj, Nstep);
        end
        
    case 'SpaTV'
       r_ops.objfun = @(a, Wx, lam) norm_mat(kdata(:) - A_for(a))^2 + lam*sum(vec(sqrt(sum(Wx.^2,4))));
        r_ops.dc = @(a) norm_mat(kdata(:) - A_for(a))^2;
        r_ops.reg = @(Wx)sum(vec(sqrt(sum(Wx.^2,4))));
        if callback
            history = iter_tv(alpha_ref,  r_ops, AHA, ksp_adj, Nstep, @callback_tv);
        else
            history = iter_tv(alpha_ref, r_ops, AHA, ksp_adj, Nstep);
        end
end
toc;

disp(' ');
disp('Reconstruction Done');

%% Project and re-scale
alpha = alpha_ref.data;
% res_a=reshape(res,dims);
im_recon=temporal_forward(alpha(:), Phi, dims);
% im = T_for(alpha);

disp('Rescaling')
alpha = alpha * scaling;
try
history.intermx = history.intermx * scaling;
end
im_recon = im_recon * scaling;


