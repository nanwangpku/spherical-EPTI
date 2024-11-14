function [B0, R2s, PD] = fit_T2sB0PD(recon, TEs, im_mask,df_pi, varargin)
%% set up images and parameters

sizes = size(recon);
%% TEs
if nargin<2
    TEs = (5+1.06*(1:sizes(end)))*1e-3;  % unit in second
    im_mask = logical(recon(:,:,1));
    df_pi = 400;
    T2s_flag = false;
    B0_flag = true;
elseif nargin<3
    im_mask = logical(recon(:,:,1));
     TEs = TEs(:).';
     df_pi = 400;
     T2s_flag = false;
     B0_flag = true;
elseif nargin<4
    df_pi = 400;
    T2s_flag = false;
     B0_flag = trues,,
elseif nargin >= 5
    TEs = TEs(:).';
    T2s_flag = ismember('R2s', varargin);
    B0_flag = ismember('B0', varargin);
elseif nargin == 4
    T2s_flag = false;
    B0_flag = true;
else
    TEs = TEs(:).';
    T2s_flag = false;
    B0_flag = true;
end
%% setup equation
N_param = 3;
recon = double(recon);
TEs = double(TEs);
TEs1 = TEs(1);
TEs = TEs-TEs(1);  % TEs is a row vector, in second
S = @(A, R2s, f0)A*exp(-TEs*R2s).*exp(1i*2*pi*f0*TEs); 
ppinv = @(x,y)x*y'/norm(x)^2;   % for normalization, x, y are row vector
opts = optimset('display', 'off');
opts.MaxFunEvals = 1000;
opts.TolX = 1e-6;

%% set up image
im_mask = im_mask>0.8;
dim_img = numel(sizes)-1;
recontemp = reshape(recon, [], sizes(end));
recontemp = recontemp(im_mask(:), :);
fits = zeros(numel(find(im_mask>0)), N_param);

%% varpro for B0 fitting
if B0_flag
clear f0
psi = exp(1i*2*pi*TEs(:)*0);
psi_pinv = pinv(psi);
psi_c = psi*psi_pinv;

% df_pi = 1/TE(1)/2; % this is the off-resonance when field-related phase shift is pi

df = -df_pi:2:df_pi; % off-resonance min f, -200 hz, max f, 200 hz, step, 2 hz

gamma = @(fb, R2s)diag(exp(1i*2*pi*fb*TEs(:)).*exp(-R2s*TEs(:)));

f0 = zeros(size(recontemp,1),1);
tic
parfor j = 1:size(recontemp,1)
    s = recontemp(j,:);
    R_search = @(fb)norm((eye(numel(TEs))-gamma(fb,0)*psi_c*gamma(-fb,0))*s(:),2);
    R0 = inf;
    f0(j) = 0;
    % calculate initial value
    for l=df
        if R_search(l)<R0
            f0(j) = l;
            R0 = R_search(l);
        end
    end
    
end
toc
f = zeros(prod(sizes(1:end-1)), 1);
f(im_mask(:)) = f0;
f = reshape(f,[sizes(1:end-1)]);
else
    f0 = zeros(size(recontemp,1),1);
    f = zeros([sizes(1:end-1)]);
end
%% mono-exponential for PD and R2*
if T2s_flag
if ~exist('f0', 'var')
    f0 = zeros(size(recontemp,1),1);
end
tic
parfor j = 1:numel(find(im_mask>0))
    curve = double(abs(recontemp(j,:).*exp(-1i*2*pi*f0(j)*TEs)));
    normcurve = curve(1);
    curve = double(curve/normcurve);
    
%     Avp = @(R2s, f0)ppinv(curve, S(1, R2s, f0));
%     temp = lsqnonlin(@(x)abs(S(Avp(x(1), x(2)), x(1), x(2))-curve), [1/0.05, 0], [1/0.3, -100], [1/0.01, 100], opts);
%     tempfit = [Avp(temp(1), temp(2))*normcurve, temp(1), temp(2)];
        
    Avp = @(R2s)ppinv(curve, S(1, R2s, 0));
    temp = lsqnonlin(@(x)abs(S(Avp(x(1)), x(1), 0)-curve), [1/0.05], [1/1], [1/0.01], opts);
%     tempfit = [Avp(temp), temp,f0(j)];
    tempfit = [Avp(temp)*normcurve/exp(-temp*TEs1), temp,f0(j)];
%     try
%     temp = lsqnonlin(@(x)abs(S(x(1), x(2), 0)-curve), [1, 1/0.05], [1, 1/1], [1, 1/0.01], opts);
%     catch
%         temp = [0 0];
%     end
%     tempfit = [temp(1)*normcurve, temp(2), f0(j)];
    fits(j,:) = tempfit;
end
toc
tempfit = fits;
fits = zeros(prod(sizes(1:end-1)), N_param);
fits(im_mask,:) = tempfit;
fits = reshape(fits, [sizes(1:end-1),N_param]);
if numel(sizes)==3
    R2s = fits(:,:,2);
    PD = fits(:,:,1);
else
    R2s = fits(:,:,:,2);
    PD = fits(:,:,:,1);
end
else
    R2s = zeros(size(f));
    PD = zeros(size(f));
end
B0 = f;
return

