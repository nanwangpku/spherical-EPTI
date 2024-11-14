function y=wave3d(x,level,wname,mask)

if nargin < 2
    level=4;
    mask = ones(size(x(:,:,1)));
elseif isempty(level)
    level=4;
end

checkLevel=true;
while checkLevel
    if sum(mod(size(x(:,:,:,1)),2^level))>0
        level=level-1;
    else
        checkLevel=false;
    end
end

if nargin < 3
    wname='db4';
elseif isempty(wname)
    wname='db4';
end

for j=1:size(x,4)
    tmp=dwt3(x(:,:,:,j),wname,'mode','per');
    tmp.mask = logical(imresize3(mask, size(tmp.dec{1}),'nearest'));
    for i = 1:2^3  % 3 is dimension
        tmp.dec{i} = tmp.dec{i}.*tmp.mask;
    end
    y(1,j) = tmp;
    for l=2:level
        tmp=dwt3(y(l-1,j).dec{1},wname,'mode','per');
        tmp.mask = logical(imresize3(mask, size(tmp.dec{1}),'nearest'));
        for i = 1:2^3  % 3 is dimension
            tmp.dec{i} = tmp.dec{i}.*tmp.mask;
        end
        y(l,j) = tmp;
    end
end

return