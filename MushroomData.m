
%-------------Split data before performing opertions on it--------------
function [mushroomSet_1,mushroomSet_2] = split_function(X)
points = size(X,1);
split = round(points*0.5);
seq = randperm(points);
mushroomTrain = X(seq(1:split),:);
mushroomDummy = X(seq(split+1:end),:);
%-----------------------------------------------------------------------

%--------------------------Finding the mean-----------------------------
function [label]=findMean(class_e,class_p,mushroomSet)
mean_e=mean(class_e,1);
mean_p=mean(class_p,1);
label=[];
points = size(mushroomSet,1)
for n=1:points
d1=pdist2(mean_e,mushroomSet(n,2:108));
d2=pdist2(mean_p,mushroomSet(n,2:108));
if d1<d2
label(n)=1;
else
label(n)=2;
end
end
%----------------------------------------------------------------------

%----------------------------Find error--------------------------------
function [error]=findError(mushroomLabel,genLabel)
error=0;
points = size(mushroomLabel,1)
for n=1:points
if mushroomLabel(n)~=genLabel(n)
error=error+1;
end
end
disp('Error is')
disp(error)
%----------------------------------------------------------------------


% Obtain train, validation and test set
[mushroomTrain,mushroomDummy] = split_function(mushroom);
[mushroomVali,mushroomTest] = split_function(mushroomDummy);

%BASELINE SYSTEM
% Minimum distance to class means classifier
class_e = mushroomTrain(1:2095,2:108);
class_p = mushroomTrain(2096:4062,2:108);
[labelVali] = findMean(class_e,class_p,mushroomVali)
[labelTest] = findMean(class_e,class_p,mushroomTest)

%%% Calculate error for validation and test set %%%
[errorVali] = findError(mushroomVali(:,1),labelVali)
[errorTest] = findError(mushroomTest(:,1),labelTest)


%USING LS CLASSIFIER
mushroomLabel(mushroomTrain(1:4062,1)==1)=0;
mushroomLabel(mushroomTrain(1:4062,1)==2)=1;

[labelVali,w]=LS((mushroomTrain(:,2:108))',(mushroomLabel(1,:)),(mushroomVali(:,2:108))');

mushroomVali_label(mushroomVali(:,1)==1)=0;
mushroomVali_label(mushroomVali(:,1)==2)=1;

%%% Calculate error for validation %%%
[errorVali] = findError(mushroomVali_label(:,1),labelVali)


%USING SVM
modelLinear=svmtrain1(mushroomTrain(:,1),mushroomTrain(:,2:108),'-c 1 -s 0 -t 0');
[labelPredict,accuracy,dec_values]=svmpredict(mushroomVali(:,1),mushroomVali(:,2:108),modelLinear);

%%% Calculate error for validation %%%
[errorVali] = findError(mushroomVali(:,1),labelPredict)


%USING KNN CLASSIFIER
labelVali = Nearest_Neighbor((mushroomTrain(:,2:108))',(mushroomTrain(:,1))', (mushroomVali(:,2:108))',1);
                             
                             
%%% Calculate error for validation %%%
[errorVali] = findError(mushroomVali(:,1),labelVali)
                             
                             
% Expanding the feature space
mushroom_expand=[];
k=109;
for x=2:108
    for y=x:108
        mushroom_expand(:,k)=mushroom(:,x).*mushroom(:,y);
        k=k+1;
    end
end