%Shows how the boolean logic 2 layer example works in section Examples and Intuitions II
% of https://www.coursera.org/learn/machine-learning/resources/RmTEz
%Result should be 1 0 0 1 i.e. both 0 or both 1

%The first part illustrates a single test case going through e.g. x1=0, x2=0
%Later on, I show how a full matrix of all 4 test cases can go through 
clc; clear;
X = [1 0 0; 1 0 1; 1 1 0; 1 1 1]; % 4 test cases
t1 = X(1,:)
t2 = X(2,:)
t3 = X(3,:)
t4 = X(4,:)

fprintf('The first column is AND, the second is OR\n');
fprintf('theta1 is a 3x2 matrix, t1 is a 1x3 vector, result is a 2x1 vector\n');
theta1 = [-30 20 20; 10 -20 -20]'
fprintf('This is NOR (exclusive OR - one or other but not both)\n');
theta2 = [-10 20 20]'

fprintf('Test case 1 \n');

a2 = t1 * theta1;
a2 = sigmoid(a2);

a2 = a2>=0.5

fprintf('Now to go through to the next layer, we add the bias term x0 = 1\n');
fprintf('So we have a 1x3 vector again\n');
fprintf('a2 gave us a 1x3 and theta2 is a 3x1 matrix so result is a 1x1 scalar\n');

% Add column of ones (the bias term x0 = 1) to a2
[m, n] = size(a2);
a2 = [ones(m, 1) a2];


a3 = theta2 * a2;
fprintf('Output ');
a3 = sigmoid(a3)>=0.5
%===
fprintf('Test case 2 \n');

a2 = t2 * theta1;
a2 = sigmoid(a2);

a2 = a2>=0.5

fprintf('Now to go through to the next layer, we add the bias term x0 = 1\n');
fprintf('So we have a 3 x 1 vector again\n');
fprintf('theta2 is a 1x3 matrix and a2 gave us a 3x1 so result is a 1x1 scalar\n');

% Add column of ones (the bias term x0 = 1) to a2
[m, n] = size(a2);
a2 = [ones(m, 1) a2];

a3 = theta2 * a2;
fprintf('Output ');
a3 = sigmoid(a3)>=0.5
fprintf('Will work the same for the other 2 test cases\n ');
%====  ALL IN ONE GO ====
fprintf('=============================\n');
fprintf('Now do it all in one go - note X already has the bias term\n');
X
theta1
theta2
fprintf('Go from input to hidden layer A2.  X * theta1\n');
A2 = X * theta1
fprintf('Sigmoidify that matrix\n');
A2 = sigmoid(A2)>=0.5
fprintf('Add column of ones - the bias term\n');

% Add column of ones (the bias term x0 = 1) to A2
[m, n] = size(A2);
A2 = [ones(m, 1) A2]

fprintf('Go from A2 to the output layer.  A2 * theta2\n');

A3 = A2 * theta2
fprintf('Sigmoidify that matrix\n');
A3 = sigmoid(A3)>=0.5
fprintf('Ta da! QED 1 0 0 1\n');
