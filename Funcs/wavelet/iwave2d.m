function y=iwave2d(x)

level=size(x,1);

for j=1:size(x,2)
    tmp = x(level,j);
    CA = tmp.dec{1,1};
    CH = tmp.dec{1,2};
    CV = tmp.dec{2,1};
    CD = tmp.dec{2,2};
    APC=idwt2(CA, CH, CV,CD, tmp.wname, 'mode',tmp.mode);
    for l=level-1:-1:1
        x(l,j).dec{1}=APC;
        tmp = x(l,j);
    CA = tmp.dec{1,1};
    CH = tmp.dec{1,2};
    CV = tmp.dec{2,1};
    CD = tmp.dec{2,2};

        APC=idwt2(CA, CH, CV,CD, tmp.wname, 'mode',tmp.mode);
    end
    y(:,:,j)=APC;
end

return