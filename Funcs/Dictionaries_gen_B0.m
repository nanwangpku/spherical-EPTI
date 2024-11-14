%%%%%%%%%%%%%%%%%%
% Dictionary Generation for 2D T2* data

function [I,B0s, OR, rfp] = Dictionaries_gen_B0(TEs,B0s)

%% Set T1 T2 B1 & Off reson.
if nargin<2
    B0s=[-450:450];  % in second;
end

rfp=[1];  % RF percentage. It's used for B1 correction
% rfp=[0.5:0.1:1.5];

OR=0;  % off-resonance effect. set  as 0
%  OR=[-800,-700,-600:50:-250,-200:20:-120,-100:10:10];
% OR=[-100:10:100];

%% Generate the dictionary
tic;
I_OR = 1;
TEs = TEs-TEs(1); % alwys start with no zero phase
for I_B0=1:numel(B0s)
    fb = B0s(I_B0);
    
    I(I_B0,I_OR,:) = exp(1i*angle(exp(1i*2*pi*fb*TEs(:))));
end

toc;
return
