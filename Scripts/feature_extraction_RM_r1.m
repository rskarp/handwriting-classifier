<<<<<<< HEAD

% Feature Extraction

close all; clearvars; clc;

L=load('emnist-letters.mat');
z=reshape( L.dataset.test.images.' ,28,28,[]);


l = L.dataset.test.labels;

N = length(z);

step = 1000;
for x = 1:step:N

    zed = z(:,:,x);
    Z = zed > 128;
    Q = regionprops(Z,'Area','BoundingBox','Centroid','Circularity','ConvexArea',...
        'ConvexHull','ConvexImage','Eccentricity','EquivDiameter','EulerNumber',...
        'Extent','Extrema','FilledArea','FilledImage','Image','MajorAxisLength',...
        'MaxFeretProperties','MinFeretProperties','MinorAxisLength','Orientation',...
        'Perimeter','PixelIdxList','PixelList','Solidity','SubarrayIdx');
    A(x) = Q.Area;
    C(1:2,x) = Q.Centroid;

    figure
    imshow(Z)
    title(l(x))

end

Areas = A(1:step:N)
Centroid = C(:,1:step:N)
=======


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
>>>>>>> 933a514726639d4d17da190c5c6e9bc45a654204
