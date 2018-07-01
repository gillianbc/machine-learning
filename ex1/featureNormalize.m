function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

%xi:=(xi−μi) / si

%Where μi is the average of all the values for feature (i) 
%and si is the range of values (max - min), or si is the standard deviation.

%Note that dividing by the range, or dividing by the standard deviation, 
%give different results. The quizzes in this course use range - the programming exercises 
%use standard deviation.

%Example: xi is housing prices with range of 100 to 2000, with a mean value of 1000. 
%Then, xi:=(price−1000)/1900.
s = size(X,1);
mu = mean(X);  % a 1 x 2 matrix of the averages of the features
% what we need is a 47 x 2 matrix so that we can subtract the average from each value in X
mu_big = ones(s,1) * mean(X);

sigma = std(X); % a 1 x 2 matrix of the standard deviations of the features
% what we need is a 47 x 2 matrix of the standard deviations of the features
sigma_big = ones(s,1) * std(X);

X_norm = (X - mu_big)./sigma_big;







% ============================================================

end
