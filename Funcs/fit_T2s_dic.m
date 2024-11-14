function [T2s_find,OR_find,p,diff] = fit_T2s_dic (I_test,I_test_p,I_dic,I,T2s,OR,down_rate,useGPU)
if ~exist('useGPU','var')
    useGPU = 1;
end

image_size=size(I_test);  % three spatial dimension + 1 time dimension
if useGPU
I_test=gpuArray(I_test);
I_dic=gpuArray(I_dic);  % T2s, OR, time points
end


down_select=image_size(1)*image_size(2)*image_size(3)/down_rate;



% p=zeros(image_size(1),image_size(2));
Nt=size(I_test,4); % time point
if useGPU
dic_find=zeros(image_size(1),image_size(2), image_size(3),'single','gpuArray');
else
    dic_find=zeros(image_size(1),image_size(2), image_size(3),'single');
end

I_2d=reshape(I,[size(I_dic,1)*size(I_dic,2),size(I_dic,3)]);
I_test_p_2d=reshape(I_test_p,[size(I_test,1)*size(I_test,2)*size(I_test,3),size(I_test,4)]);

I_dic_2d=reshape(I_dic,[size(I_dic,1)*size(I_dic,2),size(I_dic,3)]);
%%
I_dic_2d=I_dic_2d.';
%%
I_test_2d=reshape(I_test,[size(I_test,1)*size(I_test,2)*size(I_test,3),size(I_test,4)]);



% clear I_test I_dic 
for down_i=1:down_rate

bg_p=(down_i-1)*down_select+1;
ed_p=down_i*down_select;
    
I_test_2d_temp=I_test_2d(bg_p:ed_p,:);
dic_test=I_test_2d_temp*I_dic_2d;



[temp_v,temp_loc]=max(dic_test,[],2);
dic_find((down_i-1)*down_select+1:down_i*down_select)=temp_loc;
diff((down_i-1)*down_select+1:down_i*down_select)=temp_v;
clear temp_v temp_loc





clear dic_test I_test_2d_temp
%down_i
end

T2s_find=T2s(mod(dic_find-1,numel(T2s))+1);
OR_find=OR(floor((dic_find-1)/numel(T2s))+1);

I_dic_p_2d=I_2d(dic_find,:);  %
I_dic_2d = I_dic_2d.';
I_dic_2d = I_dic_2d(dic_find,:);
%% calculate pd 
for down_i=1:down_rate
% p = (sum(abs(I_test_p_2d).^2,2)).^0.5./((sum(abs(I_test_2d).^2,2)).^0.5)./((sum(abs(I_dic_p_2d).^2,2)).^0.5./((sum(abs(I_dic_2d).^2,2)).^0.5));
bg_p=(down_i-1)*down_select+1;
ed_p=down_i*down_select;
p_tmp(bg_p:ed_p)=(sum(abs(I_test_p_2d(bg_p:ed_p,:)).^2,2)).^0.5./((sum(abs(I_dic_p_2d(bg_p:ed_p,:)).^2,2)).^0.5);
end
% p=(sum(abs(I_test_p_2d).^2,2)).^0.5./((sum(abs(I_dic_p_2d).^2,2)).^0.5);
% p = abs(I_test_p_2d(:,1))./abs(I_dic_p_2d(:,1));
p=reshape(p_tmp,[image_size(1),image_size(2),image_size(3)]);
%%
diff=reshape(diff,[image_size(1),image_size(2),image_size(3)]);
%%
T2s_find = gather(T2s_find);
OR_find = gather(OR_find);
p = gather(p);
diff = gather(diff);

end
