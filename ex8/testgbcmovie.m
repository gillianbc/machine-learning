clc; clear; close all;
params = [ 1:14 ] / 10
Y = magic(4);
Y = Y(:,1:3)  %first 3 columns
R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5    % R is logical
num_users = 3
num_movies = 4
num_features = 2
lambda = 0;
%J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
%----
X = reshape(params(1:num_movies*num_features), num_movies, num_features)
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features)

J = sum(sum((((X * Theta') - Y).^2).*R)/2)
%----
fprintf('expected J =  311.63\n');

 
c = ((X * Theta') - Y).*R
X_grad = c * Theta
Theta_grad = c' * X

lambda = 6;  % now for a dollop of regularisation
p = lambda/2*sum(sum(Theta.^2))
q = lambda/2*sum(sum(X.^2))

xreg = lambda .* X
thetareg = lambda .* Theta
X_grad = X_grad + xreg
Theta_grad = Theta_grad + thetareg
%d = sum(sum(c))