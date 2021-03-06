function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

X=[ones(m,1) X];
%feed to hidden layer 1
a2=X*Theta1'; %(5000 X 401)(401 X 25) => 5000 X 25
%we should append a column for the bias  (theta(in middle)= right X left+1
%append 1's column because theta2=10*26

a2=sigmoid(a2);
a2=[ones(size(a2,1),1) a2];

out=a2*Theta2'; %(5000 X 26)(26 X 10) => 5000 X 10
h=sigmoid(out);
size(out)
[i maxIndices]=max(h,[],2);
p=maxIndices;
% =========================================================================


end
