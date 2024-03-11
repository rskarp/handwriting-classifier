function [z]=bayes_classifier(m,S,P,X)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [z]=bayes_classifier(m,S,P,X)
% Bayesian classification rule for c classes, modeled by Gaussian
% distributions (also used in Chapter 2).
%
% INPUT ARGUMENTS:
%   m:      lxc matrix, whose j-th column is the mean of the j-th class.
%   S:      lxlxc matrix, where S(:,:,j) corresponds to
%           the covariance matrix of the normal distribution of the j-th
%           class.
%   P:      c-dimensional vector, whose j-th component is the a priori
%           probability of the j-th class.
%   X:      lxN matrix, whose columns are the data vectors to be
%           classified.
%
% OUTPUT ARGUMENTS:
%   z:      N-dimensional vector, whose i-th element is the label
%           of the class where the i-th data vector is classified.
%
% (c) 2010 S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras
% Modified by Riley Karp 3/11/2024 to use mvnpdf() instead of custom
% comp_gauss_dens_val.m function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[l,c]=size(m);
[l,N]=size(X);

for i=1:N
    for j=1:c
        p = mvnpdf(X(:,i),m(:,j),S(:,:,j));
        if isnan(p)
            p = 0;
        end
        t(j)=P(j)*p;            
    end
    [num,z(i)]=max(t);
end
