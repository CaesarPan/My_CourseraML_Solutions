function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% % Set up some useful variables
% C_choice = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% sigma_choice = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% CLen = length(C_choice);
% sigmaLen = length(sigma_choice);
% error = zeros(CLen, sigmaLen);
% 
% % Pass through every combination of C & sigma
% for i = 1:CLen
%     for j = 1:sigmaLen
%         % Train the svm model
%         model = svmTrain(X, y, C_choice(i), @(x1, x2)gaussianKernel(x1, x2, sigma_choice(j)));
%         
%         % Conduct predictions
%         predictions = svmPredict(model, Xval);
%         
%         % Compute the prediction error
%         error(i, j) = mean(double(predictions ~= yval));
%     end
% end
% 
% % Find the minimum error
% [colMin, rowNum] = min(error);    % Minima of each column & their row numbers
% [totalMin, colNum] = min(colMin); % Minima of the whole matrix & its column number
% 
% C = C_choice(rowNum(colNum));     % Corresponding C
% sigma = sigma_choice(colNum);     % Corresponding sigma



%The optimal value obtained from the above code
C = 1;
sigma = 0.1;

% =========================================================================

end
