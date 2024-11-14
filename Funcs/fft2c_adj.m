function res = fft2c_adj(x)
fctr = size(x,1)*size(x,2);
% res = zeros(size(x));

size_x = size(x);

x = reshape(x, size_x(1), size_x(2), []);

% for ii=1:size(x,3)
res = sqrt(fctr)*fftshift(fftshift(ifft2(ifftshift(ifftshift(x,1),2)),1),2);
% end

res = reshape(res, size_x);

end