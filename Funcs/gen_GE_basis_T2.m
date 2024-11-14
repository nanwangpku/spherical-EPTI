function [U, U2] = gen_GE_basis_T2(N, ETL, TEs, T2vals, TEsadj)
% Generate basis, Nan, 20240109, add asjustable subspace
%
% Inputs:
%  N -- maximum number of T2 signals to simulate
%  ETL -- echo train length
%  T0 -- initial echoes time
%  TE (s) -- echo spacing
%  T2vals (s) -- array of T2 values to simulate
%  TEsadj (s) -- array of TE that actually performs
%  verbose -- print verbose output
%
% Outputs:
%  U -- temporal basis based on PCA
%  X -- [T, L] matrix of simulated signals

% randomly choose T2 values if more than N are given
if length(T2vals) > N
    idx = randperm(length(T2vals));
    T2vals = T2vals(idx(1:N));
end


LT1 = length(T2vals);

X0 = zeros(ETL, LT1);

for ii=1:LT1
    R2 = 1/T2vals(ii);
    X0(:,ii)=exp(-TEs(:)*R2);
end
X0=reshape(X0,ETL,[]);
[U, s, v] = svd(X0, 'econ');

if exist('TEsadj','var')  % same svd but on a different TE values
    ETLadj = numel(TEsadj);
    X0 = zeros(ETLadj, LT1);
    for ii=1:LT1
        R2 = 1/T2vals(ii);
        X0(:,ii)=exp(-TEsadj(:)*R2);
    end
    X0=reshape(X0,ETLadj,[]);
    U2 = X0*v*pinv(s);
else
    U2 = U;
end


end