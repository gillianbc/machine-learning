%test simple dataset to illustrate the costFunction in action
clear; clc; close all;
fprintf('Example:  3 houses.  First has 3 floors and 7 beds.\n');
X = [3,7;3,4; 6,8]
%add in a column of 1s for x0

X = [ones(size(X), 1) X]
fprintf('0 if not sold, 1 if sold\n');
y = [0;0;1]
fprintf('Starting with theta as all zeros, this means a house with no floors and no beds\n');
theta = [0;0;0]


[J,grad] = costFunction(theta,X,y);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('Total cost i.e. margin of error %f \n', J);

fprintf('If we use a different theta - say 1, 1, 1\n');
theta = [1;1;1]
[J,grad] = costFunction(theta,X,y);
fprintf('Gradient at initial theta (ones): \n');
fprintf(' %f \n', grad);
fprintf('Total cost i.e. margin of error.  It is worse: %f \n', J);
fprintf('So what theta would be best? fminunc can tell us the best theta\n');
options = optimset('GradObj', 'on', 'MaxIter', 400);
initial_theta = [0;0;0];
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
 theta
 cost
 fprintf('Let''s just prove that by plugging this new theta in\n');
 [J,grad] = costFunction(theta,X,y);
fprintf('Gradient using fminunc''s theta: \n');
fprintf(' %f \n', grad);
fprintf('Total cost i.e. margin of error %f \n', J);
fprintf('So what if we had a house with 6 floors and 9 rooms?\n  What''s the probability it would sell?\n');
fprintf('The one with 6 floors and 8 rooms didn''t sell, so I wouldn''t expect this one to either\n');
prob = sigmoid([1 6 9] * theta)
p = predict(theta, X);
p
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Maybe my dataset is too small, but it also told me that one with 3 floors and 6 rooms is even less likely to sell\n');
prob = sigmoid([1 3 6] * theta)