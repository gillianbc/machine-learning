clc; clear; close all;
fprintf('===================\n');

X = [[1 1 1]' magic(3)]
y = [7 6 5]';
theta = [0.1 0.2 0.3 0.4]'
pause;
[J calculated_g] = linearRegCostFunction(X, y, theta, 0);

fprintf('lambda 0 expected J 1.3533\n');
J
calculated_g
expected_grad = [-1.4000;
   -8.7333;
   -4.3333;
   -7.9333]
   fprintf('===================\n');
   
[J g] = linearRegCostFunction(X, y, theta, 7)
fprintf('lambda 7 expected J 1.6917\n');
J
g
expected_grad = [-1.4000; -8.2667;-3.6333;-7.0000]

