function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

[r,c] = size(X);
fprintf('rows X: %f\n', r);
fprintf('columns X: %f\n', c);
[r,c] = size(Theta1);
fprintf('rows Theta1: %f\n', r);
fprintf('columns Theta1: %f\n', c);
[r,c] = size(Theta2);
fprintf('rows Theta2: %f\n', r);
fprintf('columns Theta2: %f\n', c);
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

layer1 = sigmoid(X * Theta1');
layer1 = [ones(m, 1) layer1];
output = sigmoid(layer1 * Theta2');
[Y,p] = max(output,[],2);







% =========================================================================


end
