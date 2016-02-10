% Separating samples based on rand function
mushroom(:,109)=randperm(8124);
mushroom_train=zeros([4062 108]);
mushroom_dummy=zeros([4062 108]);
for n=1:8124
    if (mod(mushroom(n,109),2)==0)
        mushroom_train(n,1:108)=mushroom(n,1:108);
    else
        mushroom_dummy(n,1:108)=mushroom(n,1:108);
    end
end
mushroom_train(all(mushroom_train==0,2),:)=[];
mushroom_dummy(all(mushroom_dummy==0,2),:)=[];
%Separate into test data and validation data
mushroom_dummy(:,109)=randperm(4062);
mushroom_test=zeros([2031 108]);
mushroom_vali=zeros([2031 108]);
for n=1:4062
    if (mod(mushroom_dummy(n,109),2)==0)
        mushroom_vali(n,1:108)=mushroom_dummy(n,1:108);
    else
        mushroom_test(n,1:108)=mushroom_dummy(n,1:108);
    end
end
mushroom_test(all(mushroom_test==0,2),:)=[];
mushroom_vali(all(mushroom_vali==0,2),:)=[];

%BASELINE SYSTEM
class_e=mushroom_train(1:2095,2:108);
class_p=mushroom_train(2096:4062,2:108);
%Finding the mean of both classes
mean_e=mean(class_e,1);
mean_p=mean(class_p,1);
label=[];
 for n=1:2031
     d1=pdist2(mean_e,mushroom_vali(n,2:108));
     d2=pdist2(mean_p,mushroom_vali(n,2:108));
     if d1<d2
         label(n)=1;
     else
         label(n)=2;
     end
 end
  for n=1:2031
     d1=pdist2(mean_e,mushroom_test(n,2:108));
     d2=pdist2(mean_p,mushroom_test(n,2:108));
     if d1<d2
         label1(n)=1;
     else
         label1(n)=2;
     end
 end
 %Calculate error for validation and test set
 error_v=0;
 for n=1:2031
     if label(n)~=mushroom_vali(n,1)
         error_v=error_v+1;
     end
 end
disp('Error for validation set is');
disp(error_v);
disp('Error percentage for validation set is'); 
disp((error_v/2031)*100);
error_t=0;
for n=1:2031
     if label1(n)~=mushroom_test(n,1)
         error_t=error_t+1;
     end
end
disp('Error for test set is');
disp(error_t);
disp('Error percentage for test set is');
disp((error_t/2031)*100);

%Using LS Classifier
mushroom_label(mushroom_train(1:4062,1)==1)=0;
mushroom_label(mushroom_train(1:4062,1)==2)=1;
[vali_label,w]=LS((mushroom_train(1:4062,2:108))',(mushroom_label(1,1:4062)),(mushroom_vali(1:2031,2:108))');
mushroom_vali_label(mushroom_vali(1:2031,1)==1)=0;

mushroom_vali_label(mushroom_vali(1:2031,1)==2)=1;

%Find error percentage
error=0;
for n=1:2031
    if vali_label(n)~=mushroom_vali_label(n)
        error=error+1;
    end
end
disp('Error is');
disp(error);
disp('Error percentage on validation set is');
disp((error/2031)*100);

%Using SVM
model_linear=svmtrain1(mushroom_train(1:4062,1),mushroom_train(1:4062,2:108),'-c 1 -s 0 -t 0');
[label_predict,accuracy,dec_values]=svmpredict(mushroom_vali(1:2031,1),mushroom_vali(1:2031,2:108),model_linear);

%Using KNN Classifier
vali_label = Nearest_Neighbor((mushroom_train(:,2:108))',(mushroom_train(:,1))', (mushroom_vali(:,2:108))',1);
error=0;
for n=1:2031
    if vali_label(n)~=mushroom_vali(n,1)
        error=error+1;
    end
end
disp('Error for validation set is');
disp(error);
disp('Error percentage for validation set is');
disp((error/2031)*100);

% Expanding the feature space
mushroom_expand=mushroom;
k=109;
for x=2:108
    for y=x:108
        mushroom_expand(:,k)=mushroom(:,x).*mushroom(:,y);
        k=k+1;
    end
end