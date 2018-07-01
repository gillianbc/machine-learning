clear ; close all; clc;
X = [ones(3,1) magic(3)]
y = [1 0 1]'
theta = [-2 -1 1 2]'

% un-regularized
[j g] = costFunction(theta, X, y);
j;
%g;
% results
%fprintf('Expected j = 4.6832\n');
%fprintf('g =\n');
%  fprintf('0.31722\n');
%  fprintf('0.87232\n');
%  fprintf('1.64812\n');
%  fprintf('2.23787\n');
% or...
fprintf('Regularised\n');
% regularized
[j g] = costFunctionReg(theta, X, y, 4);

j
g
% results
fprintf('Expected j =  8.6832\n');
fprintf('g =\n');

 fprintf('  0.31722\n');
 fprintf(' -0.46102\n');
 fprintf('  2.98146\n');
  fprintf(' 4.90454\n');