function [R] = rotate(alpha, beta, gamma, theta)
% alpha: rotate about x
% beta: rotate about y
% theta: rotate about z
if nargin == 1
    R = [cos(alpha), -sin(alpha);
           sin(alpha), cos(alpha)];
elseif nargin == 2 || nargin == 3
    if nargin == 2
        gamma = 0;
    end
Rx = [1,      0,      0;
      0,  cos(alpha), -sin(alpha);
      0,  sin(alpha), cos(alpha)];
  
Ry = [cos(beta),  0,   sin(beta);
      0,          1,      0;
      -sin(beta), 0,   cos(beta)];
  
Rz = [cos(gamma), -sin(gamma), 0;
      sin(gamma),  cos(gamma), 0;
      0,            0,         1];
 R = Rz*Ry*Rx;
else
    a = cos(alpha);
    b = cos(beta);
    c = cos(gamma);
    d = cos(theta);
    R = [a -b -c -d; b a -d c; c d a -b; d -c b a]/sqrt(2);
end
 return