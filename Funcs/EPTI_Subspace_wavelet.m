function [im_recon,alpha,history]=EPTI_Subspace_wavelet(kdata,mask_sample,smap,Phase_total,Phi,r_ops,Func,Nstep, a_init,callback)
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
    Func = 'wavelet';
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
ns = size(Phase_total,5); % number of shot with different B0
kdata=permute(kdata,[2 3 4 1]);
mask_sample=permute(mask_sample,[2 3 4 1 5]);

K=size(Phi,2);

% Subspace basis projection
dims=[nx,ny,K];
prep = @(x)reshape(x, nx,ny,[]);
vec = @(x)x(:);

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
end
AHA = @(a) A_adj(A_for(a));
AHA_lsqr = @(a,lFlag)A_adj(A_for(a));

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
%%  wavelet setup
wlev = 4;  % wavelet level
wname = 'sym4'; % wavelet name
padmat = ceil([nx ny]/2^wlev)*2^wlev-[nx ny]; % each dimension need to be divisible by 2^wlet, which is 16 if wlet=4

if sum(padmat)~=0 % if padding
    pad = @(x)padarray(prep(x),[padmat 0],0, 'post');
    crop = @(x)x(1:nx, 1:ny,:);
    W = @(x,p)wave2d(pad(x.*exp(-1i*p)),wlev, wname, r_ops.mask);
    Wh = @(x,p)vec(crop(iwave2d(x))).*exp(1i*p(:));
else
    W = @(x,p)wave2d(prep(x.*exp(-1i*p)),wlev, wname, r_ops.mask);
    Wh = @(x,p)vec(iwave2d(x)).*exp(1i*p(:));
end
WhW = @(x,p)vec(x);
figmake = @(WX)wavelet_figmake(WX);

objfun = @(a, Wx, lam) norm_mat(kdata(:) - A_for(a))^2 + lam*sum(abs(vec(wavelet_collect(Wx))));
r_ops.dc = @(a) norm_mat(kdata(:) - A_for(a))^2;
r_ops.reg = @(Wx)sum(abs(vec(wavelet_collect(Wx))));

x = a_init ./ scaling;
p = angle(x);
% p = zeros(size(x));
Wx = W(x,p);
im0 = complex(figmake(Wx));

lambda = r_ops.lambda;
rho = r_ops.rho;
max_iter = r_ops.iter;
alpha = lambda/rho;
% define admm variable u = y/rho;
u = wavelet_applyfun(@(x)zeros(size(x)),Wx);
z = wavelet_applyfun(@(x)zeros(size(x)),Wx);

ABSTOL = 1e-4;
RELTOL = 1e-2;

abserr = sqrt(numel(ksp_adj)) * ABSTOL;

fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    'lsqr iters', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
tic
for ii = 1:max_iter
%     figure(100),imshow([log(complex(im0)), log(complex(figmake(Wx)))],[]),drawnow;
%     figure(101),imshow(abs(10*[im0,figmake(Wx)]/max(im0(:)))),drawnow
    x_old = x;
    z_old = z;
    z = wavelet_applyop(@plus, Wx, u);
    z = wavelet_applyop(@times, wavelet_applyfun(@(x)sign(x),z), wavelet_applyfun(@(x)max(abs(x)-alpha,0),z)); % soft threshold
    u = wavelet_applyop(@plus, u, wavelet_applyop(@minus, Wx,z));
    [x,~,~,nitr] = pcg(@(x)AHA(x)+rho/2*WhW(x,p), vec(ksp_adj)+rho/2*Wh(wavelet_applyop(@minus,z,u),p),...
            [],median([ii, 5, max_iter]), [],[],x(:));
    % record

    p = angle(x);
%     p = zeros(size(x));
    Wx = W(x,p);
    history.objval(ii) = objfun(x, Wx, lambda);
    history.lsqr_nitr(ii) = nitr;
    history.r_norm(ii) = norm_mat(wavelet_collect(wavelet_applyop(@minus,Wx,z)));
    history.s_norm(ii) = norm_mat(-rho * wavelet_collect(wavelet_applyop(@minus,z, z_old)));
    history.eps_pri(ii) = abserr + RELTOL * max(norm(wavelet_collect(Wx)), norm(wavelet_collect(z)));
    history.eps_dual(ii) = abserr + RELTOL * norm_mat(rho * wavelet_collect(u));
    if Nstep>1 && mod(ii, Nstep)==0
        history.intermx(:,round(ii/Nstep)) = x(:);
    end
    history.dc = r_ops.dc(x);
    history.reg = r_ops.reg(Wx);
    fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', ii, ...
        sum(history.lsqr_nitr), history.r_norm(ii), history.eps_pri(ii), ...
        history.s_norm(ii), history.eps_dual(ii), history.objval(ii));

end

toc;
disp(' ');
disp('Reconstruction Done');

%% Project and re-scale
alpha = x;
% res_a=reshape(res,dims);
im_recon=temporal_forward(alpha(:), Phi, dims);
% im = T_for(alpha);

disp('Rescaling')
alpha = alpha * scaling;
try
history.intermx = history.intermx * scaling;
end
im_recon = im_recon * scaling;


