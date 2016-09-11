function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   paramet;ers for the neural network are "unrolled" into the vector
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X = [ones(m, 1) X];                 %Add a column of 1 to the left of X. Now X is a 5000*401 matrix.
z2 = Theta1 * X';                   %z2 is a 25*5000 matrix.
a2 = sigmoid(z2);
a2 = [ones(1, size(a2, 2)) ; a2];   %a2 now is a 26*5000 matrix.
z3 = Theta2 * a2;                   %z3 is a 10*5000 matrix.
a3 = sigmoid(z3);                   %a3 is the output of the network.
a3 = a3';                           %a3 now is a 5000*10 matrix.
yMatrix = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);     %Now yMatrix is a 5000*10 matrix, each row consists of one '1' and 0s.

%Unregularized cost function
J = sum(sum(-yMatrix.*log(a3) - (1-yMatrix).*log(1-a3)));
J = J / m;

%Regularized cost function
Theta1NoBias = Theta1(:, 2:end);    %25*400
Theta2NoBias = Theta2(:, 2:end);    %10*25
J = J + (lambda*(sum(sum(Theta1NoBias.^2)) + sum(sum(Theta2NoBias.^2)))) / (2*m);

% % Part 2: Implement the backpropagation algorithm to compute the gradients
% %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
% %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% %         Theta2_grad, respectively. After implementing Part 2, you can check
% %         that your implementation is correct by running checkNNGradients
% %
% %         Note: The vector y passed into the function is a vector of labels
% %               containing values from 1..K. You need to map this vector into a 
% %               binary vector of 1's and 0's to be used with the neural network
% %               cost function.
% %
% %         Hint: We recommend implementing backpropagation using a for-loop
% %               over the training examples if you are implementing it for the 
% %               first time.
% %

delta3 = zeros(size(y));

tri_delta_1 = 0;
tri_delta_2 = 0;

delta3t = a3 - yMatrix;                  %5000*10
delta2t = (delta3t*Theta2).*((sigmoidGradient([ones(1, size(z2, 2)); z2]))');    %5000*26
delta2t_noBias = delta2t(:, 2:end);      %5000*25
tri_delta_1 = delta2t_noBias'*X;         %25*401
tri_delta_2 = (a2*delta3t)';             %10*26
Theta1_grad = tri_delta_1 / m;
Theta2_grad = tri_delta_2 / m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda*Theta1NoBias)/m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda*Theta2NoBias)/m;

% a1 = [ones(m, 1) X];
% a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
% sig = sigmoid(a2 * Theta2');
% yVector = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
% cost = -yVector .* log(sig) - (1 - yVector) .* log(1 - sig);
% 
% Theta1NoBias = Theta1(:, 2:end);
% Theta2NoBias = Theta2(:, 2:end);
% %J = (1 / m) * sum(sum(cost)) + (lambda / (2 * m)) * (sum(sum(Theta1NoBias .^ 2)) + sum(sum(Theta2NoBias .^ 2)));
% 
% delta1 = zeros(size(Theta1));
% delta2 = zeros(size(Theta2));
% 
% for t = 1:m,
% 	a1t = a1(t,:)';
% 	a2t = a2(t,:)';
% 	sigt = sig(t,:)';
% 	yVectorT = yVector(t,:)';
% 
% 	d3t = sigt - yVectorT;
% 
% 	z2t = [1; Theta1 * a1t];
% 	d2t = Theta2' * d3t .* sigmoidGradient(z2t);
% 
% 	delta1 = delta1 + d2t(2:end) * a1t';
% 	delta2 = delta2 + d3t * a2t';
% end;
% 
% Theta1ZeroBias = [ zeros(size(Theta1, 1), 1) Theta1NoBias ];
% Theta2ZeroBias = [ zeros(size(Theta2, 1), 1) Theta2NoBias ];
% Theta1_grad = (1 / m) * delta1 + (lambda / m) * Theta1ZeroBias;
% Theta2_grad = (1 / m) * delta2 + (lambda / m) * Theta2ZeroBias;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
