function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;

error_sqr = (h - y).^2;

J = sum(error_sqr) / (2 * m);

grad = (X' * (h - y))/m;

%Regularisation
factor = lambda / (2 * m);
shorttheta = theta(2:end,1:end);
reg = factor * sum(sum(shorttheta .^ 2));
J = J + reg;

reg2 = [0;((lambda / m) * shorttheta)];  %Zero for j =0
grad = grad + reg2;
% =========================================================================

grad = grad(:);

end
