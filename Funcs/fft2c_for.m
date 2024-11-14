function res = fft2c_for(x)
fctr = size(x,1)*size(x,2);
% res = zeros(size(x));

size_x = size(x);

x = reshape(x, size_x(1), size_x(2), []);

res = 1/sqrt(fctr)*fftshift(fftshift(fft2(ifftshift(ifftshift(x,1),2)),1),2);

res = reshape(res, size_x);

end





