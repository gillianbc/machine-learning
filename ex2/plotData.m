function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
% Plot Examples
%So pos is a vector of 1s and 0s indicating where the positive values are in the results set y
%X(pos,1) is the exam1 scores that led to a positive result
%X(pos,1) is the exam2 scores that led to a positive result
%We show X(pos,1) against X(pos,2) on the graph as black crosses (k+)
%Similarly, we show X(neg,1) and X(neg,2) on the graph as black circles (ko) filled yellow (y)
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
     'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
     'MarkerSize', 7);








% =========================================================================



hold off;

end
