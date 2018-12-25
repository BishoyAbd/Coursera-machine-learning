function perm =permutations (sigma_c)

%sigma_c must be a column vector (m x 1)
% I am trying to make all possible combinations of c and sigma
%sigma c is like [.01;.03 , .034] and many element 
i=1;
j=1;
num_perm=size(sigma_c,1)^2;
num_opts=size(sigma_c,1);
%initi perm of ones and size (n x 2) col1 each row contains a value for c and a value for sigma 
perm=ones(num_perm,2);
while(i<=num_perm && j<=8)
    
g=sigma_c(j)*ones(num_opts,1); % pick first value and pair it with the other values
c(i:i+num_opts-1,:)=[g sigma_c]; % (eg. form  1:8, 9:16 ... and so on)

i=i+num_opts; % step 8 each time
j=j+1;
end;
perm=c;
end