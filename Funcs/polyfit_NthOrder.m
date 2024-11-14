function [I_fitted,I_full] = polyfit_NthOrder(I,mask,Order)
%polyfit the part of the image 'I' that is inside the mask

matrix(:, 1) = ones(sum(mask(:)), 1);
FullMatrix(:, 1) = double(ones(size(mask,1)*size(mask,2),1));
[Nx, Ny,~] = size(I);
sizes = size(I);
[Xmat,Ymat]=ndgrid(-Nx/2:Nx/2-1,-Ny/2:Ny/2-1);

for CurrentOrder = 1:Order
    for Xpower = 0:CurrentOrder
            matrix = [matrix, Xmat(mask == 1).^Xpower .*Ymat(mask == 1).^(CurrentOrder-Xpower) ];
            FullMatrix = [FullMatrix, Xmat(:).^Xpower .*Ymat(:).^(CurrentOrder-Xpower)];
    end
end

%%%% old methods
% I_fitted = zeros(size(I));
% for c = 1:size(I,3)
%     if mod(c,5)==0
%         disp(num2str(c))
%     end
%     temp = I(:,:,c);
%     Ivec = temp(mask==1);
% 
%     coefs = matrix \ Ivec; % determine coefficients based on data in W2
%     I_fittedVec = FullMatrix * coefs;
%     I_fitted(:,:,c) = reshape(I_fittedVec,size(I,1),size(I,2));
% end
%%%% end old methods

%%% new methods
[polyterm,s,v]=svd(matrix,'econ');
% polyterm = polyterm*s;
% polyterm = matrix;
Ivec = I(mask==1);
w=polyterm'*Ivec (:);
%  norm(polyterm*w-pd(:),1)
y=zeros(size(Ivec(:)));
rho=1/max(abs(polyterm*w-Ivec(:)));
for it=1:10
    res=polyterm*w-Ivec(:);
    z=res+y/rho;
    z=sign(z).*max(abs(z)-1/rho,0);
    y=y+rho*(res-z);
    rho=rho*1.25;
    w=polyterm'*(z+Ivec(:)-y/rho);
    %     norm(polyterm*w-pd(:),1)
end
bias = zeros(size(I));
bias(mask)=polyterm*w;
bias = reshape(bias, sizes);
I_fitted = bias;

polyterm_full = FullMatrix*v*pinv(s);
I_full = polyterm_full*w;
I_full = reshape(I_full,sizes);


%%%%% end new methods

% [~,~,v]=svd([(polyterm*w), Ivec(:)],'econ');
% w=w*v(2)/v(1);


% I_fitted = reshape(FullMatrix*v*pinv(s)*w,Nx,[]);

% % norm(polyterm*w-pd(:),2)
% w = pcg(polyterm'*polyterm, polyterm'*Ivec, 1e-6, 20, [],[],w);
% bias = zeros(size(I));
% bias(mask)=polyterm*w;
% bias = reshape(bias, sizes);
% I_fitted = bias;
%     
    
end
