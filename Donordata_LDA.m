clc
clearvars -except donor donorrawdatamedmean
X = donorrawdatamedmean;

% Split testing data before performing opertions on it
numPoints = size(X,1);
split = round(numPoints*0.85);
seq = randperm(numPoints);
donor = X(seq(1:split),:);
donorTest = X(seq(split+1:end),:);
[row_num,col_num]=size(donor);

% SMOTE Algorithm
feat = col_num-1; % = Total number of features
ind = find(donor(:,1) == 1);
sampleC1 = size(ind,1); % = Total samples of minority class
samplesMinor = donor(ind,2:end); % array of original class samples
N = 2;
% Just to keep in rhythm with the algorithm
if (N<100)
    sampleC1 = (N/100)*sampleC1;
    N = 100;
end
newArray = [];
k = 10; % Finding 10 nearest neighbors
for i=1:sampleC1
    IDX = knnsearch(samplesMinor,samplesMinor(i,:),'k',k);
    dummyArray = Synthesize(samplesMinor,feat,N,k,i,IDX);
    newArray = [newArray;unique(dummyArray,'rows')];
end
%----------------------Synthesize function---------------------------
function [newArray] = Synthesize(samplesMinor,feat,N,k,i,IDX)
index = 1;
newArray = zeros([N,feat]);
while N~=0
    num = randi([1 k],1); % Randomly select a number from 1:k
    for f = 1:feat
        dif = samplesMinor(IDX(1,num),f) - samplesMinor(i,f);
        gap = random('uniform',0,1);
        newArray(index,f) = samplesMinor(i,f) + gap*dif;
    end
    index = index+1;
    N = N-1;
end
end
%--------------------------------------------------------------------

% Append NewArrray to existing donor dataset
donor = [donor;ones(size(newArray,1),1) newArray];
[row_num,col_num] = size(donor);

% Finding mean for each class
vec1=[];
vec2=[];
for i=1:row_num
    if(donor(i,1)==0)
        vec1(i,1:60) = donor(i,2:61);
    elseif(donor(i,1)==1)
        vec2(i,1:60) = donor(i,2:61);
    end
end
vec1(all(vec1==0,2),:)=[];
vec2(all(vec2==0,2),:)=[];
mu1=mean(vec1,1)';
mu2=mean(vec2,1)';
mu_all=mean(donor(:,2:61))';

% Covariance matrices S1 and S2
SW_1=(size(vec1,1)-1)*cov(vec1);
SW_2=(size(vec2,1)-1)*cov(vec2);

% Lets check our answer
% S_1 = zeros([60 60]);
% for i=1:size(vec1,1)
%      S_1 = S_1 + ((vec1(i,:)'-mu1)*((vec1(i,:)'-mu1)'));
% end
% RESULT: S_1==SW_1    

% Within scatter matrix
SW = SW_1 + SW_2;

% Between scatter matrix
N1 = size(vec1,1);
N2 = size(vec2,1);

SB_1 = N1*((mu1-mu_all)*(mu1-mu_all)');
SB_2 = N2*((mu2-mu_all)*(mu2-mu_all)');
SB = SB_1 + SB_2;

% Finding eigenvalues and eigenvectors
[V,D] = eig((inv(SW))*SB); %columns are the corresponding eigenvectors
[eigValues,I] = sort(diag(D), 'descend');

% Transforming the samples onto the new subspace Y=X*W where X is n×d-dimensional matrix and Y is
% the transformed n×k-dimensional matrix with the n samples projected into the new subspace).
% Using SVM with RBF kernel
% folds = 3; %cross-validation folds
acc = zeros([19 110]);

for i=2:5
    W = V(:,I(1:i));
    donor_lda = donor(:,2:61) * W;
    % grid search, and cross-validation 
    [C,gamma] = meshgrid(-5:2:15, -15:2:3); 
    d= 2; % Radial basis kernel
    for j=1:numel(C)   
        para = ['-q -v 5 -t 2 -c ', num2str(2^C(j)), ' -g ', num2str(2^gamma(j))];
        cv_acc = svmtrain(donor(:,1),donor_lda,para);
        acc(i-1,j) = cv_acc;
    end   
end

% Lets train our SVM model on the final classifier
[~,I] = max(acc(:));
[I_row, I_col] = ind2sub(size(acc),I); % I_row corresponds to number of optimum features and I_col represents best parameter value
donor_lda = donor(:,2:61) * V(:,I(1:(I_row+1)));
para = ['-q -t 2 -c ', num2str(2^C(I_col)), ' -g ', num2str(2^gamma(I_col))];
model = svmtrain(donor(:,1),donor_lda,para);
donorTest_lda = donorTest(:,2:61) * V(:,I(1:(I_row+1)));
[Testlabel,accuracy,dec_values]=svmpredict(donorTest(:,1),donorTest_lda,model);
