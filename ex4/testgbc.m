%testgbc scrap area
%% Setup the parameters you will use for this exercise
clear; clc; close all;
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
lambda = 1;  %as per ex4           
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');                         
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('Loading Data ...\n')

load('ex4data1.mat');
m = size(X, 1);
%make a permutation matrix of the y values - row wise e.g. 0100000000 is 2
%This is odd syntax, see https://www.gnu.org/software/octave/doc/interpreter/Creating-Permutation-Matrices.html
y_matrix = eye(num_labels)(y,:) ;

% Add column of ones (the bias term x0 = 1) to a2
[m, n] = size(X);
a1 = [ones(m, 1) X];

%z2 = Theta1 * a1';
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2short = a2;

% Add column of ones (the bias term x0 = 1) to a2
[m, n] = size(a2);
a2 = [ones(m, 1) a2];

%z3 = Theta2 * a2';
z3 = a2 * Theta2';
a3 = sigmoid(z3);



%Cost function
% 5000 examples, 10 classifications
%a3 is 10x5000  z3 is 10x5000
A = -1 * y_matrix' * log(a3);  %10x5000 * 5000x10 = 10x10
B = (1 - y_matrix') ;         %=> 10x5000
C = log(1 - a3);        %5000x10

%tmp = B * C;
J = (A - (B * C))/m  
%We need the sum of the diagonal terms.  Why?  Read this: https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
%element multiply by the 10x10 identity matrix
eye10 = eye(10);

J = sum(sum(J .* eye(10)))


%Back prop
%2: δ3 or d3 is the difference between a3 and the y_matrix. The dimensions are the same 
%as both, (m x r) (m training examples, r output classifications = 5000 x 10
d3 = a3 - y_matrix;

%4: δ2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product 
%of d3 and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2. 
%The size is (m x r) ⋅ (r x h) --> (m x h). The size is the same as z2, as must be.
shortTheta2 = Theta2(:,2:end);   %10x25

d2 = d3 * shortTheta2; 
gprime2 = sigmoidGradient(shortTheta2); %25x5000
d2 = d2 .* gprime2;

%all_delta = all_delta + sum(sum(d2))

%5: Δ1 or Delta1 is the product of d2 and a1. 
%The size is (h x m) ⋅ (m x n) --> (h x n)
%h = the number of units in the hidden layer - NOT including the bias unit i.e. 25
%n = the number of training features, including the initial bias unit.
Delta1 = d2' * a1; %25x5000 * 5000x401

%6: Δ2 or Delta2 is the product of d3 and a2. 
%The size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1]) = 10x26
%r = the number of output classifications
Delta2 = d3' * a2;

%7: Theta1_grad and Theta2_grad are the same size as their 
%respective Deltas, just scaled by 1/m.
Theta1_grad = Delta1 ./ m;  %Should this be element-wise or ordinary mult?
Theta2_grad = Delta2 ./ m;  %Should this be element-wise or ordinary mult?

factor = lambda / (2*m);
%the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
%the triple sum simply adds up the squares of all the individual Θs in the entire network.
all_delta = sum(sum(d3))
shortTheta1 = Theta1(:,2:end);
thetasums = sum(sum(shortTheta1.^2)) + sum(sum(shortTheta2.^2))
thetasums * factor
J = J + (thetasums * factor)
%all_delta = factor * (all_delta + thetasums)
%J = J + all_delta
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);
%t1 = factor .* (theta.^2);%here???

