function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1); 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
y_matrix = eye(num_labels)(y,:); 

%FORWARD PROPOGATION
% Add column of ones (the bias term x0 = 1)
[m1, n1] = size(X);
a1_b = [ones(m1, 1) X];  %I'm using _b where there's an included bias column
z2 = (Theta1 * a1_b')';
a2 = sigmoid(z2);


% Add column of ones (the bias term x0 = 1) to a2
[m2, n2] = size(a2);
a2_b = [ones(m2, 1) a2];
z3 = (Theta2 * a2_b')';
a3 = sigmoid(z3);


%Cost 
A = -1 * y_matrix' * log(a3);  
B = (1 - y_matrix') ;         
C = log(1 - a3);        

J = (A - (B * C))/m;  


J = sum(sum(J .* eye(num_labels)));
%fprintf('J unregularised is %f\n',J);

factor = lambda / (2*m);
shortTheta1 = Theta1(:,2:end);  %ignore the first column which is for the bias term
shortTheta2 = Theta2(:,2:end);  %ignore the first column which is for the bias term
thetasums = sum(sum(shortTheta1.^2)) + sum(sum(shortTheta2.^2));
J = J + (factor * thetasums);
%fprintf('J regularised is %f\n',J);

%BACK PROPOGATION
%Layer 3
d3 = a3 - y_matrix;

%Layer 2
gprime_z2 = sigmoidGradient(z2);
f = d3 * Theta2;
f = f(:,2:end);  %remove bias elements
d2 = f .* gprime_z2 ;
Delta2 = d3' * a2_b;

%Layer 1 
Delta1 = d2' * a1_b;
%don't actually need these bits
%gprime_z1 = sigmoidGradient(Theta1)
%d1 = d2' * Theta1'  
%f = d2 * Theta1
%f = f(:,2:end)  %remove bias elements

%Unregularised gradients
Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;
%Regularisation see Page 3 of lecture notes
reg1 = lambda * Theta1 / m;
reg1(:,1) = 0; %get rid of lambda thing for j=0
reg2 = lambda * Theta2 / m;
reg2(:,1) = 0; %get rid of lambda thing for j=0
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
