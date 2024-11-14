function [B0, T2s, PD] = fit_T2sB0PD_all(temp,TEs,mask, dic_flag,Phi,df_pi,down_rate, useGPU, varargin)

% the input temp need to be three spatial dimension + 1 temporal dimension.
% If it is not 3D, then put 1 instead
T2s_flag = ismember('T2s', varargin);
B0_flag = ismember('B0', varargin);
sizes = size(temp);
dim = numel(sizes)-1; % the last dimension of sizes is temp
if dim == 1
    temp = reshape(temp,sizes(1), 1,1,sizes(2));
elseif dim == 2
    temp = reshape(temp,sizes(1), sizes(2),1,sizes(3));
end
sizes = size(temp);
dim = numel(sizes)-1;

if B0_flag
    if exist('dic_flag','var') && dic_flag == 1
        [I,B0s, OR, rfp] = Dictionaries_gen_B0(TEs,[-df_pi:df_pi]);
        I_dic = I;  % this one is suppose to be the matrix norm one, but here no need because the norm for all curve is same
        Img = exp(-1i*angle(temp)).*mask;  % take the conjugate
        Img = reshape(Img,sizes(1),sizes(2),sizes(3),[]);
        Img_norm = Img;  % this one is suppose to be the matrix norm one,
        [B0, ~,~,~] =   fit_B0_dic(Img_norm, Img, I_dic,  I, B0s,OR,down_rate,useGPU);
        B0 = B0.*mask;
        
    else
        [B0, ~, ~] = fit_T2sB0PD(temp(:,:,:,1:6), TEs(1:6), mask, df_pi, 'B0');% using only one direction
    end
else
    B0 = zeros(sizes(1:end-1));
    
end

if T2s_flag
    if exist('dic_flag','var') && dic_flag == 1
        
        [I, T2s, OR, rfp] = Dictionaries_gen_T2s(TEs);
        I = reshape(reshape(I,[],sizes(4))*(Phi.')',size(I,1),size(I,2),[]); % match on coef maps
        I_dic=matrix_norm(I);
        
        Img = reshape(temp,[],sizes(4))*(Phi.')';
        Img = reshape(Img,sizes(1),sizes(2),sizes(3),[]);
        Img_norm = matrix_norm(Img);
        Img_norm(isnan(Img_norm)) = 0;
        Img_norm(isinf(Img_norm)) = 0;
        [T2s, OR_I, PD_map,diff] = fit_T2s_dic(Img_norm, Img, I_dic, I, T2s,OR,down_rate,useGPU);
        PD=PD_map.*(abs(diff));
        
        
    else
        
        [~, R2s,PD] = fit_T2sB0PD(temp(:,:,round(len/9):st:round(len/9*8)), TEs(round(len/9):st:round(len/9*8)), squeeze(emaps(nx,:,:)),0,'R2s');
        
        
    end
else
    T2s = zeros(sizes(1:end-1));
    PD = zeros(sizes(1:end-1));
end
