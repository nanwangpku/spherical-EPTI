
function B0_new = B0_update_new(B0_init, TEs, kspace_s, mask_sample,SEs_slc,T2svals,im_mask,ScanType,r_B0p, useGPU, useReg,dic_flag,phase_diff)
% test the B0 smoothing
% iterative, poly nomial outside
% 20230912
addpath(genpath('/local_mount/space/ladyyy/data/users/Nan/code/Funcs/Phase_Smooth_library'))
% r_B0p: parameters needed for fitting
if ~exist('useReg','var')
    useReg = 0;
end
if ~exist('dic_flag','var')
    dic_flag = 0;
end
if ~exist('phase_diff','var')
    phase_diff = zeros(size(B0_init(:,:,1)));
end

iter_B0 = r_B0p.iter_B0;
df_pi = r_B0p.df_pi;
L0 = r_B0p.L0_0;
B0vals = r_B0p.B0vals_0;
N_iteration=r_B0p.N_iteration;  % iterations within each resolver
poly_flag = r_B0p.poly; % do poly fitting or not

Nt = numel(TEs);
nt_to_fit = 10:round(Nt*3/5); % echo used to estimate B0
[Nt, Ny, Nz,Ncoils] = size(kspace_s);

ns = size(B0_init,3); % number of shot with different B0

% phs = zeros(Ny,Nz,Nx,Nt,ns);
% for ii_bin = 1:ns
%     phs(:,:,:,:,ii_bin) = exp(1i*2*pi*B0_init(:,:,:,ii_bin).*reshape(TEs,1,1,1,[])).*squeeze(exp(1i*phase_diff));
% end

% phs = exp(1i*2*pi*squeeze(B0_init).*reshape(TEs,1,1,[])).*squeeze(exp(1i*phase_diff));

% gen subspace with B0 using L0, B0vals, T2svals
N = 256; % maximum number of unique T2s values for training
[Phi0, ~] = gen_GE_basis_T2B0(N, Nt, TEs, T2svals, B0vals);
Phi_B0 = Phi0(:,1:L0);


kspace_c = kspace_s;

kspace_c(:,:,[1:round(Nz/5),round(Nz/5*4):Nz],:,:) = 0;
a0=zeros(Ny, Nz, L0);

lambda = 0;   % tikhonov regularization

down_rate = 1;
for ii_down = 10:20
    if mod(Ny*Nz,ii_down)==0
        down_rate = ii_down
        break
    end
end

disp('Reconstruction Start');
if exist('useGPU','var') && useGPU
    %     delete(gcp('nocreate'));
    %     parpool('local',2);
    kspace_c = gpuArray(single(kspace_c));
    mask_sample = gpuArray(single(mask_sample));
    Phi_B0 = gpuArray(single(Phi_B0));
    % phs = gpuArray(single(phs));
    SEs_slc = gpuArray(single(SEs_slc));
    a0 = gpuArray(single(a0));
end

% loop a few times for B0
for ii = 1:iter_B0

    phs = zeros(Ny,Nz,Nt,1,ns);
    for ii_bin = 1:ns
        phs(:,:,:,1,ii_bin) = exp(1i*2*pi*B0_init(:,:,ii_bin).*reshape(TEs,1,1,1,[])).*squeeze(exp(1i*phase_diff));
    end
    if exist('useGPU','var') && useGPU
        phs = gpuArray(single(phs));
    end
    tic;
    [~,a_B0]=EPTI_Subspace_Tik(kspace_c,mask_sample,squeeze(SEs_slc),phs,Phi_B0,a0,N_iteration,lambda,zeros(size(a0(:))));
    toc;
    
%     POCS_flag = 0;
%     kdata = fft2c(reshape(a_B0,Ny,Nz,L0));
%     a_B0 = lowRes_PhaseSmooth(kdata, [Ny,Nz,round(Nz*1/2)],POCS_flag);

    temp = reshape(reshape(a_B0,[],L0)*Phi_B0.',Ny,Nz,[]);
    
  
    temps = gather(temp.*phs);
       

    % if strcmp(ScanType, 'Cir') || strcmp(ScanType, 'BKCR')
    len = size(Phi_B0,1);
    st = ceil(len/80); % step size
    if dic_flag
        [I,B0s, OR, rfp] = Dictionaries_gen_B0(TEs(round(len/9):st:round(len/2)),[-df_pi:df_pi]);
        I_dic = I;  % this one is suppose to be the matrix norm one, but here no need because the norm for all curve is same
        for ii_bin = 1:ns
        Img = exp(-1i*angle(temps(:,:,round(len/9):st:round(len/2),ii_bin))).*im_mask;  % take the conjugate
        Img = reshape(Img,Ny,Nz,1,[]);
        Img_norm = Img;  % this one is suppose to be the matrix norm one,
        [B0_new(:,:,ii_bin), ~,~,~] =   fit_B0_dic(Img_norm, Img, I_dic,  I, B0s,OR,down_rate);
        B0_new(:,:,ii_bin) = B0_new(:,:,ii_bin).*im_mask;
        end

    else
        [B0_new, ~,~] = fit_T2sB0PD(temps(:,:,round(len/9):st:round(len/2)), TEs(round(len/9):st:round(len/2)), im_mask,df_pi,'B0');
    end
    
    
    if poly_flag
%         B0_init = B0_new;
%         B0_init = imgaussfilt(B0_new);
        for ii_bin = 1:ns
        mask_img = abs(temps(:,:,round(st),ii_bin));
        [Nbin, Edge] = histcounts(abs(mask_img(:)));
        cw = Edge(find(cumsum(Nbin)>0.999*prod(size(mask_img)),1));

        mask_img = mask_img>cw/30;

        bias = polyfit_NthOrder(B0_new(:,:,ii_bin)-B0_init(:,:,ii_bin),mask_img,10);

        B0_init(:,:,ii_bin) = B0_init(:,:,ii_bin)+bias;
        end
    else
        B0_init = B0_new;
    end
  
    
%     POCS_flag = 0;
%     a_tmp = abs(temps(:,:,round(st+len/2))).*exp(1i*B0_init*mean(diff(TEs))/2);  % ass as a small phase with TE = mean(diff(TEs))
%     kdata = fft2c(reshape(a_tmp,Ny,Nz));
%     a_smooth = lowRes_PhaseSmooth(kdata, [Ny,Nz,round(Nz*3/5)],POCS_flag);  % Nz/2 is for partial Fourier removal
%     B0_init = angle(a_smooth)/(mean(diff(TEs))/2);

end

% after the initial loop, regenerate B0 dict and phase

B0vals = r_B0p.B0vals_end;  % in Hz
L0 = r_B0p.L0_end; % subspace size
N = 256; % maximum number of unique T2s values for training
[Phi0, ~] = gen_GE_basis_T2B0(N, Nt, TEs, T2svals, B0vals);
Phi_B0 = Phi0(:,1:L0);

% phs = exp(1i*2*pi*squeeze(B0_init).*reshape(TEs,1,1,[])).*squeeze(exp(1i*phase_diff));
phs = zeros(Ny,Nz,Nt,1,ns);
for ii_bin = 1:ns
    phs(:,:,:,1,ii_bin) = exp(1i*2*pi*B0_init(:,:,ii_bin).*reshape(TEs,1,1,1,[])).*squeeze(exp(1i*phase_diff));
end
if exist('useGPU','var') && useGPU
    phs = gpuArray(single(phs));
    Phi_B0 = gpuArray(single(Phi_B0));
end

tic;
    [~,a_B0]=EPTI_Subspace_Tik(kspace_c,mask_sample,squeeze(SEs_slc),phs,Phi_B0,a0,N_iteration,lambda,zeros(Ny*Nz*L0,1));
toc;

if useReg
    % % LLR
    r_ops.iter = 15;
    r_ops.lambda = .03;
    r_ops.rho = r_ops.lambda*2;
    for ii_divisor = 9:-1:5
        if mod(Ny,ii_divisor)==0
            divisor(1) = ii_divisor;
            break;
        end
    end
    for ii_divisor = 9:-1:5
        if mod(Nz,ii_divisor)==0
            divisor(2) = ii_divisor;
            break;
        end
    end
    r_ops.block_dim = [divisor(1),divisor(2)];
    r_ops.max_iter = 20;
    r_ops.tol = 1e-6;
    [~,a_B0,~]=EPTI_Subspace_ADMM(kspace_c,mask_sample,squeeze(SEs_slc),phs,Phi_B0,r_ops,'LLR', 1, a_B0, false);
end


disp('Reconstruction Done');
if exist('useGPU','var') && useGPU
    kspace_c = gather(kspace_c);
    Phi_B0 = gather(Phi_B0);
    phs = gather(phs);
    mask_sample = gather(mask_sample);
    a_B0 = gather(a_B0);
    a0 = gather(a0);
end

% POCS_flag = 0;
% kdata = fft2c(reshape(a_B0,Ny,Nz,L0));
% a_B0 = lowRes_PhaseSmooth(kdata, [Ny,Nz,round(Nz*1/2)],POCS_flag);

temp = reshape(reshape(a_B0,[],L0)*Phi_B0.',Ny,Nz,[]);

% B0 smooth
% POCS_flag = 0;
% kdata = fft2c(reshape(temp,Ny,Nz,[]));
% temp = lowRes_PhaseSmooth(kdata, [Ny,Nz,Nz],POCS_flag);  % Nz*3/5 is for partial Fourier removal


temps = temp.*phs;
    
% POCS_flag = 0;
% kdata = fft2c(reshape(temps,Ny,Nz,[]));
% temps = lowRes_PhaseSmooth(kdata, [Ny,Nz,Nz],POCS_flag);  % Nz*3/5 is for partial Fourier removal

% if strcmp(ScanType, 'Cir') || strcmp(ScanType, 'BKCR')
len = size(Phi_B0,1);
st = ceil(len/80); % step size
if dic_flag
    [I,B0s, OR, rfp] = Dictionaries_gen_B0(TEs(round(len/9):st:round(len/2)),[-df_pi:df_pi]);
    I_dic = I;  % this one is suppose to be the matrix norm one, but here no need because the norm for all curve is same
    for ii_bin = 1:ns
        Img = exp(-1i*angle(temps(:,:,round(len/9):st:round(len/2),ii_bin))).*im_mask;  % take the conjugate
        Img = reshape(Img,Ny,Nz,1,[]);
        Img_norm = Img;  % this one is suppose to be the matrix norm one,
        [B0_new(:,:,ii_bin), ~,~,~] =   fit_B0_dic(Img_norm, Img, I_dic,  I, B0s,OR,down_rate);
        B0_new(:,:,ii_bin) = B0_new(:,:,ii_bin).*im_mask;
    end

else
    [B0_new, ~,~] = fit_T2sB0PD(temps(:,:,round(len/9):st:round(len/2)), TEs(round(len/9):st:round(len/2)), im_mask,df_pi,'B0');
end

% mask_img = abs(temps(:,:,round(st)));
% [Nbin, Edge] = histcounts(abs(mask_img(:)));
% cw = Edge(find(cumsum(Nbin)>0.999*prod(size(mask_img)),1));
% 
% mask_img = mask_img>cw/30;
% 
% bias = polyfit_NthOrder(B0_new-B0_init,mask_img,10);
% 
% B0_new = B0_init+bias;
% B0_new = imgaussfilt(B0_new);
%     figure, imagesc(B0_new, [-50 50]), axis('image'), colormap(jet)
clear temps kspace_c SEs_slc
return