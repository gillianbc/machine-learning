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
%For j=0, we don't want to do any lambda malarkey
%We want the cost of first column of X using first element of theta
%i.e. 118x1 and 1x1
%For all the others, we want all except the first column of X and all except
%first element of theta i.e. 118x27 and 27x1 and we then do some lambda stuff to it
%The lambda/m * theta term is 118x27

%Get first column of X
Xzero = X(:,1);
%Get first element of theta
Qtheta = theta(1);
%Get the cost for J0
[Jzero,gradzero] = costFunction(Qtheta,Xzero,y);
%=============
%Get all but the first column of X
Xothers = X(:,2:end);
%Get all except first element of theta
Otheta = theta(2:end);
%Get the cost for J all others
%temp
%[Jothers, gradothers] = costFunction(Otheta,Xothers,y);
xtheta = X * theta;  % 118x1 vector
h = sigmoid(xtheta); % 118x1 vector

A = -1 * y' * log(h); %y' is a 1x118 vector, log(h) is a 118x1 so A is just 1x1 i.e. a scalar
B = (1 - y)' ;        % 1x118 vector
C = log(1 - h);       % 118x1 vector


J = (A - (B * C))/m;  %B * C is 1x118*118*1 = a scalar so J is a scalar

%===  now the gradient - vectorised implementation on page 4 of lecture notes - ignore alpha
grad = (X' * (h - y))/m;



%regfactor = ((lambda / m) * Xothers);
%Jothers = Jothers .+ regfactor;
%J = Jzero .+ Jothers;



% =============================================================

end
