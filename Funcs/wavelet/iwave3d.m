function y=iwave3d(x)

level=size(x,1);

for j=1:size(x,2)
  APC=idwt3(x(level,j));
  for l=level-1:-1:1
    x(l,j).dec{1}=APC;
    APC=idwt3(x(l,j));
  end
  y(:,:,:,j)=APC;
end

return