function Y=wavelet_figmake(Wx)

j=size(Wx,1);
wsl=floor(size(Wx(j,1).dec{1},3)/2+1);
for k=1:size(Wx,2)
  Y(:,:,k)=[Wx(j,k).dec{1}(:,:,wsl), Wx(j,k).dec{1,2,1}(:,:,wsl); Wx(j,k).dec{2,1,1}(:,:,wsl), Wx(j,k).dec{2,2,1}(:,:,wsl)];
end

for j=size(Wx,1)-1:-1:1
  wsl=floor(size(Wx(j,1).dec{1},3)/2+1);
  Yold=Y;
  clear Y;
  for k=1:size(Wx,2)
    Y(:,:,k)=[Yold(:,:,k), Wx(j,k).dec{1,2,1}(:,:,wsl); Wx(j,k).dec{2,1,1}(:,:,wsl), Wx(j,k).dec{2,2,1}(:,:,wsl)];
  end
end

Y=sum(abs(Y),3);

return;