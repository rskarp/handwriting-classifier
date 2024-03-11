
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
