function output=matrix_norm(matrix_a,dim_a)

if nargin < 2
   temp=size(matrix_a);
   dim_a=length(temp);
   clear temp
end

%norm_temp=max(matrix,[],dim_a);
norm_temp=(sum(abs(matrix_a).^2,dim_a)).^0.5;

repmat_matrix=ones(1,dim_a);
repmat_matrix(dim_a)=size(matrix_a,dim_a);

output=matrix_a./repmat(norm_temp,repmat_matrix);



end

