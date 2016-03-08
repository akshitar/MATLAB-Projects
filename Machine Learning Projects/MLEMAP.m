% Given a linear Gaussian model with E~N(0,0.25) derive and plot the
% Maximum likelihood estimate. Further given the prior derive and also plot
% the Maximum Aposterior of w

%%----FUNCTION PLOT_PLANE----%%
function [] = plot_plane( w, color )
%plot_plane Summary Plots a plane with given normal vector and color
if size(w) ~= [3,1]
	disp('Error: input normal has to be a 3 by 1 vector');
	return
end
hold on;
xlim = 3*[-1 -1;1 -1;1 1 ;-1 1];
ylim = [xlim ones(4,1)]*w;
patch(xlim(:,1),xlim(:,2),ylim,color);
alpha(0.3);
end

%%----SUBQUESTION 1----%%
%Expression for MLE for w

x_train = x(1:30,:);
y_train = y(1:30,:);
xT_train = (x_train)';
w_mle = inv(xT_train*x_train)*xT_train*y_train;

%%----SUBQUESTION 2----%%
%Expression for MAP for w

term1 = xT_train*x_train;
term3 = xT_train*y_train;
tau = 0.001; sigma = 0.5;
vector = repmat(1/tau,[3 1]);
variance_inv = diag(vector);
term2 = sigma*variance_inv;
mean_w = repmat(sigma,[3 1]);
term4 = term2*mean_w;
term5 = inv(term1+term2);
w_map = term5*(term3+term4);

%%----SUBQUESTION 3----%%
%Plotting the graph

x1 = x_train(:,1);
x2 = x_train(:,2);
scatter3(x1,x2,y_train,'filled'); hold on;
plot_plane(w_mle,'g');
plot_plane(w_map,'r');
title('Plot showing training data, wmle plane and wmap plane');
xlabel('x1'); ylabel('x2'); zlabel('y'); hold on;
legend('Training data','MLE plane','MAP plane');


%%----SUBQUESTION 4----%%
% MSE for MLE using testing data
dummy_1=0; dummy_2=0;
for i=31:550
    dummy_1=x(i,:)*w_mle;
    dummy_2=dummy_2+(y(i)-dummy_1).^2;
end
mle_mse=(1/520)*dummy_2;

% MSE for MAP using testing data
dummy_1=0; dummy_2=0;
for i=31:550
    dummy_1=x(i,:)*w_map;
    dummy_2=dummy_2+(y(i)-dummy_1).^2;
end
map_mse=(1/520)*dummy_2;


%%----SUBQUESTION 5----%%
%Expression for MAP for w
term1= xT_train*x_train;
term3= xT_train*y_train;
tau= 0.2; sigma= 0.5;
vector= repmat(1/tau,[3 1]);
variance_inv= diag(vector);
term2= sigma*variance_inv;
mean_w= repmat(sigma,[3 1]);
term4= term2*mean_w;
term5= inv(term1+term2);
w_map= term5*(term3+term4);

%Plotting the graph
x1=x_train(:,1);
x2=x_train(:,2);
scatter3(x1,x2,y_train,'filled'); hold on;
plot_plane(w_mle,'g');
plot_plane(w_map,'r');
title('Plot showing training data, wmle plane and wmap plane');
xlabel('x1'); ylabel('x2'); zlabel('y'); hold on;
legend('Training data','MLE plane','MAP plane');

% MSE for MLE using testing data
dummy_1=0; dummy_2=0;
for i=31:550
    dummy_1=x(i,:)*w_mle;
    dummy_2=dummy_2+(y(i)-dummy_1).^2;
end
mle_mse=(1/520)*dummy_2;

% MSE for MAP using testing data
dummy_1=0; dummy_2=0;
for i=31:550
    dummy_1=x(i,:)*w_map;
    dummy_2=dummy_2+(y(i)-dummy_1).^2;
end
map_mse=(1/520)*dummy_2;


%%----SUBQUESTION 6----%%
%Expression for MAP for w
term1= xT_train*x_train;
term3= xT_train*y_train;
tau= 0.001; sigma= 0.5;
vector= repmat(1/tau,[3 1]);
variance_inv= diag(vector);
term2= sigma*variance_inv;
mean_w= repmat(sigma,[3 1]);
term4= term2*mean_w;
term5= inv(term1+term2);
w_map= term5*(term3+term4);

%Plotting the graph
x1=x_train(:,1);
x2=x_train(:,2);
scatter3(x1,x2,y_train,'filled'); hold on;
plot_plane(w_mle,'g');
plot_plane(w_map,'r');
title('Plot showing training data, wmle plane and wmap plane');
xlabel('x1'); ylabel('x2'); zlabel('y'); hold on;
legend('Training data','MLE plane','MAP plane');

% MSE for MLE using testing data
dummy_1=0; dummy_2=0;
for i=201:550
    dummy_1=x(i,:)*w_mle;
    dummy_2=dummy_2+(y(i)-dummy_1).^2;
end
mle_mse=(1/350)*dummy_2;


% MSE for MAP using testing data
dummy_1=0; dummy_2=0;
for i=201:550
    dummy_1=x(i,:)*w_map;
    dummy_2=dummy_2+(y(i)-dummy_1).^2;
end
map_mse=(1/350)*dummy_2;
