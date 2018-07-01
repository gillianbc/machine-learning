function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

%Look back at the lecture where he added and subtracted a little bit to 
%get the gradient of a tangent to the curve

%first calculate the sigmoid of the matrix
gsigmoid = 1.0 ./ (1.0 + exp(-z));

%From the lecture notes, The g-prime derivative terms can also be written out as:
%g′(z(l))=a(l) .∗ (1−a(l))
%https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm

g = gsigmoid .* (1 - gsigmoid);


%test case is here: https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/oRjv1jr2EeWm9SIAC5HAew










% =============================================================




end
