function Y=wavelet_applyfun(func,WU)
Y=WU;
for j=1:size(WU,1)
  for l=1:numel(WU(j,1).dec) %include all APCs
    for k=1:size(WU,2)
      Y(j,k).dec{l}=func(WU(j,k).dec{l});
    end
  end
end

return