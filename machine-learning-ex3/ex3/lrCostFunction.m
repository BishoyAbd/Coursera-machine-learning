function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%=======================================================
%J unregiolized= J=1/m * sum((sigmoid(X*theta) -y))
%========================================================
%J regulized=J=1/m * sum(((sigmoid(X*theta) - y) + lambda/(2*m)*sum(theta.^2 )))

% =============================================================
%gradient unregulaizd=>
%X->m*n
%h=X*theta (m X 1)
%B=h-y (still m X 1 )
%grad =(1/m) X'*B  ( 400 X 5000 * 5000 X 1 ) =(400 X 1) 
% =============================================================
% =============================================================
%gradient regulized=>
%X->m*n
%h=X*theta (m X 1)
%regu_term=lambda/m *thtea  ( n * 1 )
%B=h-y + (still mX1 ) 
%grad(2) =(1/m) X'*B + regu_term ( 400 X 5000 * 5000 X 1 ) =(400 X 1) 
% =============================================================


grad = grad(:);

h=sigmoid(X*theta); 
regu_term=lambda/m *theta(2:end) ;
B=h-y  ;
%grad(2) =(1/m) X'*B + regu_term ( 400 X 5000 * 5000 X 1 ) =(400 X 1) 
grad(1)=1/m *(h-y)' *X(:,1);
grad(2:end)=1/m* X(:,2:end)'*B+ regu_term;

h=sigmoid(X*theta); 
J=1/m *( -y'*log(h) - (1-y)'* log(1-h)) + lambda/(2*m)*sum(theta(2:end).^2 );
%note that (-y'*log(h)) is a scaler and ((1-y)'* log(1-h)) is a scalar...
%subtracting produces another scalar that is added to sum(thtea(2:end).^2)
%or else use element wise
% J=1/m * sum(( -y.*log(h) - (1-y).* log(1-h))) + lambda/(2*m)*sum(theta(2:end).^2 );

end
