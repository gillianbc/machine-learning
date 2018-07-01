function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices
%   i.e. nn_params is Big Theta and we want the individual theta1 and theta2 
%   matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%===============
% GBC - why are we doing this function?  Answer: To find Big Theta
% If we have this function that finds J and grad, 
% we can then use this function as an argument in the fminunc function which
% will give us the best Theta matrices.  Once we know what the best Theta matrices are,
% we can use them with an input matrix to accurately predict the outcome.
%===============
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1))

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1))

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%make a permutation matrix of the y values - row wise e.g. 0100000000 is 2
%This is odd syntax, see https://www.gnu.org/software/octave/doc/interpreter/Creating-Permutation-Matrices.html
y_matrix = eye(num_labels)(y,:) ;

%FORWARD PROPOGATION
% Add column of ones (the bias term x0 = 1) to a2
[m, n] = size(X);
X = [ones(m, 1) X];

z2 = Theta1 * X'
a2 = sigmoid(z2)

% Add column of ones (the bias term x0 = 1) to a2
[m, n] = size(a2');
a2 = [ones(m, 1) a2']

z3 = Theta2 * a2'
a3 = sigmoid(z3)

fprintf('nnCostFunc paused. Press enter to continue.\n');
pause;
%Cost 
% 5000 examples, 10 classifications
% a3 is 10x5000  z3 is 10x5000
A = -1 * y_matrix' * log(a3');  %10x5000 * 5000x10 = 10x10
B = (1 - y_matrix') ;         %=> 10x5000
C = log(1 - a3');        %5000x10


J = (A - (B * C))/m  %B*C is 10x5000 * 5000x10 = 10x10 so J is a 10x10 matrix

%We need the sum of the diagonal terms.  Why?  Read this: https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
%element multiply by the 10x10 identity matrix
%This gives us the unregularised cost
J = sum(sum(J .* eye(num_labels)));

%We now need to add the regularisation term to J as per ex4.pdf page 6
factor = lambda / (2*m);
shortTheta1 = Theta1(:,2:end);  %ignore the first column which is for the bias term
shortTheta2 = Theta2(:,2:end);  %ignore the first column which is for the bias term
thetasums = sum(sum(shortTheta1.^2)) + sum(sum(shortTheta2.^2));
J = J + (factor * thetasums)

%BACK PROPOGATION
%Layer 3
d3 = a3 - y_matrix'
%Layer 2
d2 = d3' * Theta2
gprime2 = sigmoidGradient(Theta2)
Delta2 = d3 * a2
fprintf('Delta2 is correct ( I don''t know how cos I''ve not called sigmoid gradient\n');
%
z2 = z2';
gz2 = sigmoidGradient(z2)
a3 = a3'
d3 = a3 - y_matrix
fprintf('d3 is correct (trial and error)\n');
d2 = d3 * shortTheta2

d1 = d2' * Theta1'
gprime1 = sigmoidGradient(Theta1)
Delta1 = d2 * X'

%d2 = d2 .* gprime2;
%Delta1 = d2' * X; %25x5000 * 5000x401
%Delta2 = d3 * a2;
%Theta1_grad = Delta1 ./ m;  %Should this be element-wise or ordinary mult?
%Theta2_grad = Delta2 ./ m;  %Should this be element-wise or ordinary mult?















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
