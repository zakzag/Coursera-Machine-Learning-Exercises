function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

a1__ = [ones(m,1) X];
z2__ = a1__ * Theta1';
a2__ = [ones(m, 1) sigmoid(z2__)];
z3__ = a2__ * Theta2';
a3__ = sigmoid(z3__);
h0__ = a3__;

y__ = zeros(m, num_labels);

% makes output vectors for each learning vectorize
% 5 => 0000100000
for i = 1:m 
  y__(i, y(i)) = 1;
endfor

% You need to return the following variables correctly 
J = 1 / m * sum(sum( -y__ .* log(h0__) .- (1.- y__) .* log(1 .- h0__)));

% regularization part. Notice Theta1(:,2:end) where theta values for bias were removed!
% Notice .^2 - each element were squared one by one, not a matrix with a matrix
reg = lambda / (2 * m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


J= J + reg;

% initial values are zeros for thetas (greek capital delta)
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% hard to accomplish it without for loop
for t = 1:m
  % input layer
  a1_ = [1 X(t,:)]';
  
  % hidden layer 
  z2_ = Theta1 * a1_;
  a2_ = [1; sigmoid(z2_)];
  % output layer
  z3_ = Theta2 * a2_;
  a3_ = sigmoid(z3_);
  % expected output
  y_ = ([1:num_labels]==y(t))';
  
  % delta at output
  delta3_ = a3_ - y_;
  
  % computing delta 
  delta2_ = Theta2' * delta3_ .* [1; sigmoidGradient(z2_)];
  delta2_ = delta2_(2:end);
  
  Theta1_grad = Theta1_grad + delta2_ * a1_';
  Theta2_grad = Theta2_grad + delta3_ * a2_';
endfor

% regularization for backpropagation
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

grad = [Theta1_grad(:); Theta2_grad(:)]
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
