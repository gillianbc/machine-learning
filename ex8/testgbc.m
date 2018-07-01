clc; close all; clear;
X = sin(magic(4));
X = X(:,1:3);
[mu sigma2] = estimateGaussian(X)

fprintf('Expected mu = -0.3978779  0.3892253  -0.0080072\n');
fprintf('Expected sigma2 = 0.27795    0.65844   0.20414\n');