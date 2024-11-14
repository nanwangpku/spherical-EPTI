function Y=wavelet_applyop(func,A,B)

if ~isstruct(A)
  disp('Error: A must be a wavelet structure')
  return
end

if isstruct(B)
  iswavelet=true;
elseif isscalar(B)
  iswavelet=false;
else
  disp('Error: unrecognized B. Must be a scalar or wavelet structure.')
  return
end

Y=A;

for j=1:size(A,1)
  for l=1:numel(A(j,1).dec) %include all APCs
    for k=1:size(A,2)
      if iswavelet
      Y(j,k).dec{l}=func(A(j,k).dec{l},B(j,k).dec{l});
      else
        Y(j,k).dec{l}=func(A(j,k).dec{l},B);
      end
    end
  end
end

return