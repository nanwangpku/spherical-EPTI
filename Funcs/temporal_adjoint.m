function [ alpha ] = temporal_adjoint(im, Phi, dims, Wt, flatten)
% temporal_adjoint orthognal projection according to:
%   alpha = Phi' im
%
%    im: [I1, I2, ..., In, T] Fully sampled image at each echo time (oracle)
%    Phi: [T, K] orthogonal basis operator
%    alpha: [I1, I2, ..., In,  K] image coefficients for basis elements
%  I1, I2, ..., In are image/coil/other dimensions
%  T is echo train length (number of echo times)
%  K is the number of basis coefficients

if nargin < 4 || nargin < 5
    flatten = false;
end
if nargin < 3
    dims = size(im);
end

[T, K] = size(Phi);
% if T==K && K ==1
%     dims(end+1) = 1;
% end
L = prod(dims(1:end-1));

dims2 = dims;
dims2(end) = K;

% compute coefficients, alpha, of orthogonal projection
coeffs =Phi' * reshape(im, L, T).';

alpha = reshape(coeffs.', dims2);

if nargin < 3 || ~exist('Wt','var')
    Wt = reshape(ones(1,dims2(end)), [ones(1,numel(dims2)-1) dims2(end)]);
end

alpha = alpha.*Wt;

if flatten
    alpha = reshape(alpha, size(coeffs));
end

alpha=alpha(:);
% alpha=real(alpha);
end
