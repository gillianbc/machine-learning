function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%X is a 118x2 matrix - first column is all 1s
%y is a 118x1 vector
%theta is a 118x28


%Get the unregularized cost J and gradient vector
[J, grad] = costFunction(theta,X,y);

%Now set first element of theta to zero
theta(1) = 0
%Now do the lambda thing for the cost => lambda / 2m * sum (theta sqrs)
J
fprintf('factor\n');
factor = lambda / (2*m)
%See section Regularized Logistic Regression in the lecture notes
%https://www.coursera.org/learn/machine-learning/resources/Zi29t

t1 = factor .* (theta.^2)

J = J + sum(t1);

%Gradient - now calculate the lambda things for gradient descent
%(lambda / m ) * theta
factor = lambda / m
t2 = factor .* theta
%now add to the unregularized gradients returned by the cost function
grad = grad + t2;

% =============================================================

end
