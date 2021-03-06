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
                 hidden_layer_size, (input_layer_size + 1)); %hiddenX401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); %10Xhidden+1
             

             
% Setup some useful variables
m = size(X, 1);
         

%forward l1->l2
X=[ones(m,1) X]; %mxn1(input features 401)
z2=X*Theta1'; %mxn X nxhidden_size(num hidden features 25)=> m x hidden
a2=sigmoid(z2);%5000x25
%forward l2->l3(out)
a2=[ones(m,1) a2]; %mxn1(hidden features 26)
z3=a2*Theta2'; %mxhiddden_size+1 * hiddden_size+1xout hidden(num hidden features 26)=>5000X10
a3=sigmoid(z3); %mx10
J=0;

%unregulize J (the exersize call the same file for both J and J regulized ,
%you could comment the J code..or just modify it)
for k=1:num_labels,
new_y= y==k;
current_a3=a3(:,k);
J = J + (1/m)* sum(-new_y.*log(current_a3)-(1-new_y).*log(1-current_a3));

end

%regulize J (the exersize call the same file for both J and J reg ...
%you could comment the above or just modify it)
% 
regulization_term_for_j=lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) +sum(sum(Theta2(:,2:end).^2)));
J = J +regulization_term_for_j;

     delta2=zeros(size(Theta1)); %25x401
     delta3=zeros(size(Theta2));  % 10x26
for i =1:m,
    % 1-feed forward
    example=X(i,:); %1x401
    z2=Theta1*example'; %25x1
    a2=sigmoid(z2); 
    
    z2=[1 ;z2]; %26x1
    a2=[1 ;a2]; %26x1
    
    z3=Theta2*a2;   %10x1
    a3=sigmoid(z3); %10x1
    
    % 2- compute sigma3
    
    y_example=zeros(size(a3));
    y_example(y(i))=1;
    fprintf("a3 %d %d  y_example %d %d",size(a3),size(y_example))
    sigma3=(a3-y_example) ;
    fprintf("\nsigma3 %d %d",size(sigma3))

    % 3- sigma 2
    
    fprintf("\nTheta2 %d %d ,sigma3  %d %d ,z2  %d %d",size(Theta2),size(sigma3),size(z2))
    sigma2=Theta2'*sigma3.*sigmoidGradient(z2); % 26x10 x 10x1 .* 26x1    
    fprintf("\nsigma2 %d %d",size(sigma2))
    % 4- accumilate
   
    fprintf("\ndelta2 %d %d simga2 %d %d ,example  %d %d ",size(delta2),size(sigma2),size(example))
    delta2=delta2+sigma2(2:end)*example;   % 25x1 x 1x401
    
   
    fprintf("\ndelta3 %d %d simga3 %d %d ,a2  %d %d ",size(delta3),size(sigma3),size(a2))
    delta3=delta3+sigma3*a2';    % 10x1 x1 x26
    
end    

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Theta1_grad=(1/m)* delta2;
Theta2_grad=(1/m)* delta3;


Theta1_grad(:,1)=(1/m)* delta2(:,1);
Theta2_grad(:,1)=(1/m)* delta3(:,1);
Theta1_grad(:,2:end)=(1/m)* delta2(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end)=(1/m)* delta3(:,2:end)+(lambda/m)*Theta2(:,2:end);


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
