function [C,sigma] = findBest_params()
%FINDBEST_PARAMS Summary of this function goes here
%   Detailed explanation goes here
% 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
sigma_c=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
perms=permutations(sigma_c);
models=ones(64,1);
errors_64=ones(64,1);
min=1000000000;
C = 1; sigma = 0.1;
good_c=0;
good_sigma=0;
for i=1:64,
     %pick row row 
    C=perms(i,1); 
    sigma=perms(i,2) ;
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); %train
    predictions=svmPredict(model,Xval); %predcit
    cv_error=mean(double(predictions ~= yval)); % get mean error
    if(cv_error<min)
        %update min and  save better c,sigma till now
        min=cv_error;
        good_c=C;
        good_sigma=sigma;
    end
end
end

