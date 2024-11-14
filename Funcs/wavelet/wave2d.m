function y=wave2d(x,level,wname,mask)

if nargin < 2
    level=4;
    mask = ones(size(x(:,:,1)));
elseif isempty(level)
    level=4;
end

checkLevel=true;
while checkLevel
    if sum(mod(size(x(:,:,1)),2^level))>0
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

for j=1:size(x,3)
    [CA,CH,CV,CD]=dwt2(x(:,:,j),wname,'mode','per');
    tmp.sizeINI = size(CA)*2;
    tmp.mode = 'per';
    tmp.wname = wname;
    tmp.mask = logical(imresize(mask, size(CA),'nearest'));
    tmp.dec{1,1} = CA.*tmp.mask;
    tmp.dec{1,2} = CH.*tmp.mask;
    tmp.dec{2,1} = CV.*tmp.mask;
    tmp.dec{2,2} = CD.*tmp.mask;
    y(1,j) = tmp;
    for l=2:level
        [CA,CH,CV,CD]=dwt2(y(l-1,j).dec{1},wname,'mode','per');
        tmp.sizeINI = size(CA)*2;
        tmp.mode = 'per';
        tmp.wname = wname;
        tmp.mask = logical(imresize(mask, size(CA),'nearest'));
        tmp.dec{1,1} = CA.*tmp.mask;
        tmp.dec{1,2} = CH.*tmp.mask;
        tmp.dec{2,1} = CV.*tmp.mask;
        tmp.dec{2,2} = CD.*tmp.mask;
        y(l,j) = tmp;
    end
end

return