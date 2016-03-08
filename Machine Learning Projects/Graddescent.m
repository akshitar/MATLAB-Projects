%Implement a steepest descent algorithm	running the script for several different step sizes. 
%Draw the path and report number of iteration for each case .	You	are	given	a	skeleton	

clear all; clc
%Standardizing the features
Xtrain_norm=standardize(Xtrain);
Xtest_norm=standardize(Xtest);

%Log-transforming the features
Xtrain_transform=log((Xtrain(:,:))+0.1);
Xtest_transform=log((Xtest(:,:))+0.1);

%Binarize the features
Xtrain_bin=heaviside(Xtrain);
Xtrain_bin(Xtrain_bin==0.5)=0;
Xtest_bin=heaviside(Xtest);
Xtest_bin(Xtest_bin==0.5)=0;

% Logitic model for normalized data
fitFn = @(Xtrain_norm, ytrain,param)...
logregFit(Xtrain_norm, ytrain, 'lambda', param(1));
predictFn = @logregPredict;
lossFn = @(ytest, yhat)mean(yhat ~= ytest);
nfolds = 2;
paramRange = [0:0.01:01]';
[model, bestParam, mu, se] = ...
fitCv(paramRange, fitFn, predictFn, lossFn, Xtrain_norm, ytrain);
yhat_train = logregPredict(model, Xtrain_norm);
yhat_test = logregPredict(model, Xtest_norm);
error_train_std = mean(ytrain~=yhat_train);
error_test_std = mean(ytest~=yhat_test);

% Logistic model for log transformed data
fitFn = @(Xtrain_transform, ytrain,param)logregFit(Xtrain_transform, ytrain,
'lambda', param(1));
predictFn = @logregPredict;
lossFn = @(ytest, yhat)mean(yhat ~= ytest);
nfolds = 2; paramRange = [0:0.01:01]';
[model, bestParam, mu, se] = fitCv(paramRange, fitFn, predictFn, lossFn,
Xtrain_transform, ytrain);
yhat_train = logregPredict(model, Xtrain_transform);
yhat_test = logregPredict(model, Xtest_transform);
error_train_trf = mean(ytrain~=yhat_train);
error_test_trf = mean(ytest~=yhat_test);

% Logitic model for binaristic data
fitFn = @(Xtrain_bin, ytrain,param)...
logregFit(Xtrain_bin, ytrain, 'lambda', param(1));
predictFn = @logregPredict;
lossFn = @(ytest, yhat)mean(yhat ~= ytest);
nfolds = 2;
paramRange = [0:0.01:01]';
[model, bestParam, mu, se] = ...
fitCv(paramRange, fitFn, predictFn, lossFn, Xtrain_bin, ytrain);
yhat_train = logregPredict(model, Xtrain_bin);
yhat_test = logregPredict(model, Xtest_bin);
error_train_bin = mean(ytrain~=yhat_train);
error_test_bin = mean(ytest~=yhat_test);

%Scatter plot
j=0;
for i=1:1536
    j=j+1;
    if yhat_test(i)==1
        Xtest_spam(j,:)=Xtest(i,:);
    else
        Xtest_nospam(j,:)=Xtest(i,:);
    end
end
Xtest_x_spam=sum(Xtest_spam(:,1:48),2);
Xtest_y_spam=sum(Xtest_spam(:,49:54),2);
scatter(Xtest_x_spam,Xtest_y_spam,'x','r'); hold on;
Xtest_x_nospam=sum(Xtest_nospam(:,1:48),2);
Xtest_y_nospam=sum(Xtest_nospam(:,49:54),2);
scatter(Xtest_x_nospam,Xtest_y_nospam,'o','y'); hold on;
title('Scatter plot');xlabel('Number of Kerywords'); 
ylabel('Number of special characters');

%Generating histogram
Xtest_1=sum(Xtest(:,1:48),2);
Xtest_2=sum(Xtest(:,49:54),2);
hist3([Xtest_1(yhat_test==1) Xtest_2(yhat_test==1)]);hold on;
title('Histogram for spam emails');xlabel('Number of Kerywords');
ylabel('Number of special characters');zlabel('Frequescy');
hist3([Xtest_1(yhat_test==0) Xtest_2(yhat_test==0)]);hold on;
title('Histogram for no-spam emails');xlabel('Number of Kerywords');
ylabel('Number of special characters');zlabel('Frequescy');