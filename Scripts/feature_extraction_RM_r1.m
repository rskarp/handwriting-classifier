

% Feature Extraction

close all; clearvars; clc;

L=load('emnist-letters.mat');
z=reshape( L.dataset.test.images.' ,28,28,[]);



l = L.dataset.test.labels;

N = length(z);

Letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];

step = 1;
showPlots= 0;


for x = 1:step:N
%if(x ~= 139 && x ~= 386 && x ~= 395 && x ~= 408 && x ~= 502)


    zed = z(:,:,x);
    Z = zed > 128;


    if showPlots
        %figure
        imshow(Z)
        title(l(x))
    end

    L = Z;
    %Z = bwconncomp(Z);


    Q = regionprops(Z,'Area','Centroid','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation','ConvexArea','Circularity',...
        'Solidity','Perimeter');

    if(length(Q) > 1)
        continue;
    end

    Q.FullImage = Z;
    Q.LetterImage = Z;
    Q.Filename = "emnist-letters.mat";
    Q.Name = Letters(l(x));
    Q.Letter = Letters(l(x));
    Q.HuMoments = hu_moments(Z);

    R(x) = Q;

  
end
