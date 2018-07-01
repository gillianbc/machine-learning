function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
% 1/m * ( A - BC ) - see bottom of page 3 of the Lecture Notes - vectorised implementation
% A column of 1s for x0 will have been prepended to X from the dataset before this cost function is called

%e.g. X is a 100x3 matrix (100 examples x constant plus 2 features),  theta is a 3x1 vector
xtheta = X * theta;  % 100x1 vector
h = sigmoid(xtheta); % 100x1 vector

A = -1 * y' * log(h); %y' is a 1x100 vector, log(h) is a 100x1 so A is just 1x1 i.e. a scalar
B = (1 - y)' ;        % 1x100 vector
C = log(1 - h);       % 100x1 vector


J = (A - (B * C))/m;  %B * C is 1x100*100*1 = a scalar so J is a scalar

%===  now the gradient - vectorised implementation on page 4 of lecture notes - ignore alpha
grad = (X' * (h - y))/m;









% =============================================================

end
