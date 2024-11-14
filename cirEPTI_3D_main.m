%% readme
%this is the example code to run sEPTI recon for the example dataset
% before this example script, the data has been corrected for gradient
% delay. The gradient delay is corrected using the standard approach
% provided by the vendors
%
% All the ramp sampling data has been gridded to Cartesian grid
% using NUFFT. So the process in the script only involves Cartesian data
%% setup path

addpath(genpath('./Funcs'))
%% use GPU
% useGPU=(exist('gpuNUFFT','file')>1); %true if using gpuNUFFT. false if using irt.
useGPU=true;
if useGPU
    try
        choose_GPU; %Give MATLAB main software access to the GPU before the gpuNUFFT software locks it up
    catch
        gpudev = gpuDevice(2);
    end
end



%% data introduction
% data to load
% kspace_cart is the kspace stored in cell format. each cell is the data
% from one shot in format: Nx, number of echoes, Ncoils

% SEs: sensitivity in Ny, Nz, Nx, Ncoils

% emaps: espirit mask, mask out the backdround in images (Ny,Nz,Nx)

% phase_diff: eddy current phase difference estimated from calibration
% data, in size Ny, Nz, Nx,Nt

% B0_init: initial B0 map in Ny,Nz,Nx from calibration data

% TEs: echo times in second, dim: (1,Nt)

% ky: dim (shot, Nt0), for a shot ii_shot, a time point ii_t, the phase
% encoding in kspace is ky(ii_shot, ii_t);

% kz: : dim (shot, Nt0), for a shot ii_shot, a time point ii_t, the
% partition encoding in kspace is kz(ii_shot, ii_t);

% kt: : dim (shot, Nt0), for a shot ii_shot, a time point ii_t, the order
% of the time in the total order.

load('sEPTI_example.mat');

Nx = 240;
Ny = 240;
Nz = 216;
Ncoils = 20;
shot = 820;  % number of shot in total
ScanType = 'BKCR'
dic_flag = 1;
%% T2* dictionary
T2svals=[5:1:100,102:3:200]*1e-3;  % in second, for dictionary generation
% gen subspace without phase
Nt = numel(TEs);
L = 3; % subspace size
Nbasis = 256; % maximum number of unique T2s values for training
[Phi0, ~] = gen_GE_basis_T2(Nbasis, Nt, TEs, T2svals);

%% generate circular mask, this is to identify the un-sampled k-space corner

disp('set up circular mask')

mask_ham = zeros(Ny,Nz);
for i = 1:shot
    for j = 1:size(kspace_cart{i},2)
        mask_ham(ky(i,j),kz(i,j)) = mask_ham(ky(i,j),kz(i,j))+1;
    end
end
mask_ham = logical(mask_ham);
se = strel('disk',3,4);
mask_ham = imclose(mask_ham, se);



%% recon


for nx = 1:Nx
    nx

    SEs_slc = SEs(:,:,nx,:);
    B0_slc = B0_init(:,:,nx);
    phase_diff_slc = phase_diff(:,:,nx,:);
    mask_img = logical(squeeze(emaps(:,:,nx))>0.9);

    Func = 'wavelet'
    r_ops.iter = 20;
    r_ops.mask = mask_img
    r_ops.lambda = .005;
    r_ops.rho = r_ops.lambda;
    r_ops.max_iter = 10;
    r_ops.tol = 1e-6;



    recon_3D_by_nx;


    B0(:,:,nx) = B0_slc;
    a0_all(:,nx) = a0(:);
    a_all(:,nx) = a(:);

    %  display
    range = 1000;
    if strcmp(ScanType, 'Cir')
        range = range*2;
    end
    temp0 = reshape(reshape(a0,[],L)*Phi.',Ny,Nz,1,[]);
    [Nbin, Edge] = histcounts(abs(temp0(:)));
    cw = Edge(find(cumsum(Nbin)>0.999*prod(size(temp0)),1))

    temp = reshape(reshape(a,[],L)*Phi.',Ny,Nz,1,[]);
    % fitting
    df_pi_T2s = 0; % this is a variable not usefull for T2*
    down_rate = 1;
    len = size(Phi,1);
    st = max(round(len/50),1); % step size
    tmpl = round(len/9):st:round(len/9*8);
    [~, T2s_rec0(:,:,nx), PD_rec0(:,:,nx)] = fit_T2sB0PD_all(temp0(:,:,:,tmpl),TEs(tmpl),permute(emaps(nx,:,:),[2 3 1]), dic_flag,Phi(tmpl,:),df_pi_T2s,down_rate,useGPU,'T2s');
    [~, T2s_rec(:,:,nx), PD_rec(:,:,nx)] =   fit_T2sB0PD_all(temp(:,:,:,tmpl), TEs(tmpl),permute(emaps(nx,:,:),[2 3 1]), dic_flag,Phi(tmpl,:),df_pi_T2s,down_rate,useGPU,'T2s');

end

%% display
h=implay(PD_rec/200);
h=implay(T2s_rec/0.15);
h.Visual.ColorMap.Map = jet;
