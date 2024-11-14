function Y=wavelet_collect(WU)
vec=@(x)x(:);

Y=complex(zeros(prod(WU(1).sizeINI),size(WU,2)));
rowindex=0;
for j=1:size(WU,1)
  for l=2:numel(WU(j,1).dec) %ignore APC
    temp=vec(WU(j,1).dec{l}); % k = 1
    temp(end,size(WU,2))=0; %preallocate
    for k=2:size(WU,2)
      temp(:,k)=vec(WU(j,k).dec{l});
    end
    Y(rowindex+(1:size(temp,1)),:)=temp;
    rowindex=rowindex+size(temp,1);
  end
end

j=size(WU,1); % last level
l=1; %APC
temp=vec(WU(j,1).dec{l}); % k = 1
temp(end,size(WU,2))=0; %preallocate
for k=2:size(WU,2)
  temp(:,k)=vec(WU(j,k).dec{l});
end
Y(rowindex+(1:size(temp,1)),:)=temp;

return