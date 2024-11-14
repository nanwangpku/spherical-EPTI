%% this code is used to recon a single nx for 3D data 

%% put kspace and sampling mask together
masks = zeros(Nt,Ny,Nz,Ncoils);  % nbins represent how many bins


kspace_slc = single(zeros(Nt,Ny, Nz, Ncoils));
mask_sample = zeros(size(kspace_slc));
for i = 1:shot
    tmp = kspace_cart{i};
    for j = 1:size(tmp,2)
        if ky(i,j)<=0 || ky(i,j)>Ny ||kz(i,j)<=0 || kz(i,j)>Nz
            continue
        end
        kspace_slc(kt(i,j),ky(i,j),kz(i,j),:) = reshape(kspace_cart{i}(nx,j,:,:),1,1,1,[]);
        masks(kt(i,j),ky(i,j),kz(i,j),:) = 1;
        mask_sample(kt(i,j),ky(i,j),kz(i,j),:) = 1; % the 5th dimension is correspond to how many B0s
    end
end


%% generate additional sampling mask to cover the un-sampled corner in the image to avoid random filling of the corner

tmpl = squeeze(find(mask_ham(:)==0));  % mask a column vector tmpl containing all the zeros location in mask_ham
[mask_ind_y,mask_ind_z] = ind2sub([Ny,Nz],tmpl);
for ii_t = 1:Nt

    tmp_v = randsample(1:numel(mask_ind_y),min(round(numel(tmpl)/Nt),numel(mask_ind_y)));
    for ii_tmp = 1:numel(tmp_v)
        masks(ii_t,mask_ind_y(tmp_v(ii_tmp)),mask_ind_z(tmp_v(ii_tmp)),:)=1;
    end
    mask_ind_y(tmp_v) = [];
    mask_ind_z(tmp_v) = []; % those who got selected will be removed

end
mask_sample = mask_sample|masks;




%% iterative B0 update

    df_pi = 450;

    % r_B0p: parameters needed for B0 update
    r_B0p.B0vals_0 = [-40:1:40];  % in Hz
    r_B0p.L0_0 = 6;   % start with how many rank
    r_B0p.B0vals_end = [-10:1:10];  % in Hz, for last iterations
    r_B0p.L0_end = 3;   % rank for last iterations
    r_B0p.df_pi = df_pi;
    r_B0p.N_iteration = 50;  % number of solver iterations
    r_B0p.iter_B0 = 2;  % number of B0 iterations
    r_B0p.poly = 0; % do polynomial fitting in B0_iter or not
    useReg = 1;

    B0_slc = B0_update_new(B0_slc,TEs, kspace_slc, mask_sample, SEs_slc, T2svals, mask_img, ScanType,r_B0p, useGPU,useReg,dic_flag);


%% final recon
L = 3;
Phi = Phi0(:,1:L);

Phi = Phi* (rotate(pi/4,pi/4,pi/4)');  % for the purpose of balance



phs = exp(1i*2*pi*squeeze(B0_slc).*reshape(TEs,1,1,[])).*squeeze(exp(1i*phase_diff_slc));



if exist('useGPU','var') && useGPU

    kspace_slc = gpuArray(single(kspace_slc));
    mask_sample = gpuArray(single(mask_sample));
    Phi = gpuArray(single(Phi));
    phs = gpuArray(single(phs));
    SEs_slc = gpuArray(single(SEs_slc));
end

[~,a0,~] = EPTI_Subspace_CG_B0var(kspace_slc,mask_sample,squeeze(SEs_slc),phs,Phi,zeros(Ny*Nz*L,1),30);


a0 = reshape(a0,Ny,Nz,[]);


[~,a,~]=EPTI_Subspace_wavelet(kspace_slc.*sum(mask_sample,5),mask_sample,squeeze(SEs_slc),phs,Phi,r_ops,Func, 1, a0, true);


if exist('useGPU','var') && useGPU
    kspace_slc = gather(kspace_slc);
    Phi = gather(Phi);
    phs = gather(phs);
    mask_sample = gather(mask_sample);
    SEs_slc = gather(SEs_slc);
    a0 = gather(a0);
    a = gather(a);
end
