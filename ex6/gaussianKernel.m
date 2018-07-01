function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

p = sum((x1 - x2).^2);
p = -1 * p / (2 * (sigma^2));
sim = exp(p);




% =============================================================
    
end
%!test
%! sim = gaussianKernel([1 2 3], [2 4 6], 3);
%! sim_expected = 0.45943;
%! 
%! assert(sim, sim_expected, 0.00001);

