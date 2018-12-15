function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

h=X*theta; % mx2 * 2x1 =>mx1,,,,,,, y=mx1, will take the transpose
h=sigmoid(h);
J = -(1/m)*sum(y'*log(h)+(1-y)'*log(1-h));
grad = zeros(size(theta));

for i=1:n,
                               
    grad(i)=(1/m)*sum((h-y)'*X(:,i));
    %(h-y) -> mx1 ... X(:,i)-> mX1 
    % so (mX1)'*(mX1)=> [1] .. no need for sum but
    % (i didnot test it without sum)
end

% Note: grad should have the same dimensions as theta
%








% =============================================================

end
