
% Feature Extraction

close all; clearvars; clc;

L=load('emnist-byclass.mat');

%L=load('emnist-letters.mat');
z=reshape( L.dataset.test.images.' ,28,28,[]);

% Initialize variables
featuresData = {};
numFeatureObjs = 0;



l = L.dataset.test.labels;

N = length(z);

Letters = ['0','1','2','3','4','5','6','7','8','9',...
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',...
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];


step = 1;
showPlots= 0;
i = 1;


for x = 1:step:N
%if(x ~= 139 && x ~= 386 && x ~= 395 && x ~= 408 && x ~= 502)

    if(l(x) < 10)
        continue;
    end

    zed = z(:,:,x);
    %Z = zed > 128;
    Z = im2bin(zed);




    L = Z;
    cc = bwconncomp(Z);

    % Combine all objects into one object
    if cc.NumObjects > 1
        pixels = [];
        for i=1:cc.NumObjects
            pixels = [pixels; cc.PixelIdxList{i}];
        end
        cc.NumObjects = 1;
        cc.PixelIdxList = {pixels};
    end


    Q = regionprops(cc,'Area','Centroid','MajorAxisLength','MinorAxisLength','Eccentricity','Orientation','ConvexArea','Circularity',...
        'Solidity','Perimeter');
    

    if(length(Q) > 1)
        continue;
    end

    Q.FullImage = Z;
    Q.LetterImage = Z;
    Q.Filename = "emnist-letters.mat";
    Q.Name = Letters(l(x)+1);
    Q.Letter = Letters(l(x)+1);
    Q.HuMoments = hu_moments(Z);


    if showPlots
        figure
        imshow(Z)
    
        title(Letters(l(x)+1))
        %title(l(x))
    end
    R(i) = Q;
    i = i + 1;
    

end

save test_features_rm.mat R
