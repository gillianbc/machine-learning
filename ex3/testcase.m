% input
fprintf('Input data\n');
clc; clear; close all;

theta = [-2; -1; 1; 2];
X = [ones(5,1) reshape(1:15,5,3)/10]
y = [1;0;1;0;1] >= 0.5       % creates a logical array
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda);
J
grad' 
fprintf('My results\n');
% results
fprintf('Expected J =  2.5348\n');
fprintf('grad = 0.14656, -0.54856,0.72472,1.39800\n');

fprintf('\nOne Vs All -------\n\n');
X = [magic(3) ; sin(1:3); cos(1:3)];
y = [1; 2; 2; 1; 3];
num_labels = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda)
%output:
fprintf('Expected all_theta =\n\n');
fprintf('  -0.559478   0.619220  -0.550361  -0.093502\n');
fprintf('  -5.472920  -0.471565   1.261046   0.634767\n');
fprintf('   0.068368  -0.375582  -1.652262  -1.410138\n');
fprintf('Predict One vs All\n');  
all_theta = [1 -6 3; -2 4 -3]
X = [1 7; 4 5; 7 8; 1 4]
p = predictOneVsAll(all_theta, X)
%output:
p
fprintf('Expected \n1\n 2\n 2\n 1\n');

fprintf('Predict =========\n');
Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);
p = predict(Theta1, Theta2, X)
% you should see this result
fprintf('Expected p = 4 1 1 4 4 4 4 2 (but vertical)\n');
  
   