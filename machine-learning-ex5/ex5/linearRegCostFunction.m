function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% X = 12x2
% theta = 2x1

h = X * theta;
J = 0;
grad = zeros(size(theta));
thetaNoZero = theta(2:end);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
J = 1 / (2 * m) * sum((h - y) .^ 2) + lambda / (2 * m) * sum(thetaNoZero .^ 2);

thetaReg = [0; thetaNoZero];

grad = (1 / m) * X' * (h - y) + lambda / m * thetaReg;

% =========================================================================

grad = grad(:);

end
