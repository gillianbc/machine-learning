clear; clc;
il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of classifications
nn = [ 1:18 ] / 10;  % nn_params  row vector of weights
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = 4;

[J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda);



%[J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda);

fprintf('Should get J = 19.474\n');
fprintf('grad 0.76614 0.97990...0.322331 etc\n');
grad
%see https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
%Shows all intermediate variables too