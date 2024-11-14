%%%%%%%%%%%%%%%%%%
% Dictionary Generation for 2D T2* data

function [I,T2s, OR, rfp] = Dictionaries_gen_T2s(TEs)

%% Set T1 T2 B1 & Off reson.
T2s=[10:0.5:100,103:3:300,310:5:1000]*1e-3;  % in second;

rfp=[1];  % RF percentage. It's used for B1 correction
% rfp=[0.5:0.1:1.5];

OR=0;  % off-resonance effect. set  as 0
%  OR=[-800,-700,-600:50:-250,-200:20:-120,-100:10:10];
% OR=[-100:10:100];

%% Generate the dictionary
tic;
I_OR = 1;
for I_T2s=1:numel(T2s)
    R2s = 1./T2s(I_T2s);
    
 I(I_T2s,I_OR,:) = exp(-TEs(:)*R2s);
end

toc;
return
