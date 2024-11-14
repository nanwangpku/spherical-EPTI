function res = fft3c(x)
fctr = size(x,1)*size(x,2)*size(x,3);

sizes = size(x);
res = zeros([size(x,1),size(x,2),size(x,3),prod(sizes(4:end))],'like',x);


for n=1:prod(sizes(4:end))
    res(:,:,:,n) = 1/sqrt(fctr)*fftshift(fftn(ifftshift(x(:,:,:,n))));
end

res = reshape(res,sizes);


