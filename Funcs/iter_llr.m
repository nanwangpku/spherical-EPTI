function [ history ] = iter_llr(x_ref, r_ops, Aop, b, Nstep, callback_fun )
%iter_admm ADMM algorithm for solving locally low rank regularization:
%
% \min_x 0.5 * || y - Ax ||_2^2 + lambda * sum_r || R_r(x) ||_*  --- (1)
%   where R_r extracts a block from x around position r
%
% Inputs:
%  x_ref -- pointer to solution to (1), with result stored in x_ref.data 
%  r_ops.rho -- rho augmented lagrangian parameter for ADMM
%  r_ops.max_iter -- maximum number of iterations
%  r_ops.objfun -- handle to objection function, J(x, sv, lambda)
% 
%  r_ops.lambda -- regularization parameter for LLR
%  r_ops.block_dim [Wy, Wz] -- block size for LLR
%
%  r_ops.max_iter -- maximum number of iterations for LSQR within ADMM
%  r_ops.tol -- tolerance
%
%  Aop -- function handle for A'*A*x, Aop(x)
%  b -- adjoint of data, A'*y
%
%  NStep -- how many intermediate x to be saved
% 
%  callback_fun -- execute  callback_fun(x) at the end of each iteration
%
% Outputs:
%  history -- struct of history/statistics from the optimization

if nargin < 8
    use_callback = false;
else
    use_callback = true;
end

x = x_ref.data;
x = reshape(x,size(b));
z = zeros(size(x));
u = zeros(size(x));

rho = r_ops.rho;
max_iter = r_ops.iter;
objfun = r_ops.objfun;

lambda = r_ops.lambda;
block_dim = r_ops.block_dim;

ABSTOL = 1e-4;
RELTOL = 1e-2;

abserr = sqrt(numel(b)) * ABSTOL;

vec = @(x)x(:);

fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
    'lsqr iters', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective');

tic;

for ii=1:max_iter
    z_old = z;
    % update z using LLR singular value thresholding
    xpu = x + u;
    [z, s_vals] = llr_thresh(xpu, lambda / rho, block_dim);    
    % update u
    u = xpu - z;
    % update x using LSQR. the operator may change if rho changes
    AHA_lsqr = @(a) (vec(rho * reshape(a, size(b))) + vec(Aop(reshape(a, size(b))))); % for lsqr
%     [a, ~, ~, lsqr_nitr] = symmlq(AHA_lsqr, b(:) + rho * (z(:) - u(:)), ...
%         r_ops.tol, r_ops.max_iter, [], [], x(:)); 
    [a, ~, ~, lsqr_nitr] = pcg(@(a)AHA_lsqr(a), b(:) + rho * (z(:) - u(:)), ...
        r_ops.tol, median([ii,5,r_ops.max_iter]), [], [], x(:)); 
    x = reshape(a, size(b));

    
    % record
    history.objval(ii) = objfun(x, s_vals, lambda);
    history.lsqr_nitr(ii) = lsqr_nitr;
    history.r_norm(ii) = norm_mat(x - z);
    history.s_norm(ii) = norm_mat(-rho * (z - z_old));
    history.eps_pri(ii) = abserr + RELTOL * max(norm(x(:)), norm(z(:)));
    history.eps_dual(ii) = abserr + RELTOL * norm_mat(rho * u);
    if Nstep>1 && mod(ii, Nstep)==0
        history.intermx(:,round(ii/Nstep)) = x(:);
    end
    history.dc = r_ops.dc(x);
    history.reg = r_ops.reg(s_vals);

    fprintf('%3d\t%10d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', ii, ...
        sum(history.lsqr_nitr), history.r_norm(ii), history.eps_pri(ii), ...
        history.s_norm(ii), history.eps_dual(ii), history.objval(ii));
    % fprintf('%3d\n', ii)
    
    if use_callback
        callback_fun(x, history, ii);
       
    end
    
    x_ref.data = x;
end
t2 = toc;

history.nitr = ii;
history.run_time = t2;



