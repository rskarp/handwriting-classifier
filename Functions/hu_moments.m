% ~~Hu Moments of order 3~~
% This function calculates hu moments acording to the formulas given in 
% http://en.wikipedia.org/wiki/Image_moment

% First of all the central moments(mu(i,j)) and center of mass(x_bar,y_bar)
% of the image are calculated according to the formulas given in the link 
% then hu moments are calculated and saved in a vector as an output

% ~~Author: Ishrat Badami~~0
% ~~Computer Graphics Department, University of Bonn~~

function  hu_moments_vector  = hu_moments( image )

[height, width] = size(image);

% define a co-ordinate system for image 
xgrid = repmat((-floor(height/2):1:ceil(height/2)-1)',1,width);
ygrid = repmat(-floor(width/2):1:ceil(width/2)-1,height,1);

[x_bar, y_bar] = centerOfMass(image,xgrid,ygrid);

% normalize coordinate system by subtracting mean
xnorm = x_bar - xgrid;
ynorm = y_bar - ygrid;

% central moments
mu_11 = central_moments( image ,xnorm,ynorm,1,1);
mu_20 = central_moments( image ,xnorm,ynorm,2,0);
mu_02 = central_moments( image ,xnorm,ynorm,0,2);
mu_21 = central_moments( image ,xnorm,ynorm,2,1);
mu_12 = central_moments( image ,xnorm,ynorm,1,2);
mu_03 = central_moments( image ,xnorm,ynorm,0,3);
mu_30 = central_moments( image ,xnorm,ynorm,3,0);


%calculate first 8 hu moments of order 3
I_one   = mu_20 + mu_02;
I_two   = (mu_20 - mu_02)^2 + 4*mu_11^2;
I_three = (mu_30 - 3*mu_12)^2 + (mu_03 - 3*mu_21)^2;
I_four  = (mu_30 + mu_12)^2 + (mu_03 + mu_21)^2;
I_five  = (mu_30 - 3*mu_12)*(mu_30 + mu_12)*((mu_30 + mu_12)^2 - 3*(mu_21 + mu_03)^2) + (3*mu_21 - mu_03)*(mu_21 + mu_03)*(3*(mu_30 + mu_12)^2 - (mu_03 + mu_21)^2);
I_six   = (mu_20 - mu_02)*((mu_30 + mu_12)^2 - (mu_21 + mu_03)^2) + 4*mu_11*(mu_30 + mu_12)*(mu_21 + mu_03);
I_seven = (3*mu_21 - mu_03)*(mu_30 + mu_12)*((mu_30 + mu_12)^2 - 3*(mu_21 + mu_03)^2) - (mu_30 - 3*mu_12)*(mu_21 + mu_03)*(3*(mu_30 + mu_12)^2 - (mu_03 + mu_21)^2);
I_eight = mu_11*((mu_30 + mu_12)^2 - (mu_03 + mu_21)^2) - (mu_20 - mu_02)*(mu_30 + mu_12)*(mu_03 + mu_21);

hu_moments_vector = [I_one, I_two, I_three,I_four,I_five,I_six,I_seven,I_eight];

end

% calculate scale invariant central moments
function cm = central_moments( image ,xnorm,ynorm,p,q)
    
    cm = sum(sum((xnorm.^p).*(ynorm.^q).*image));
    cm_00 = sum(sum(image)); %this is same as mu(0,0);
    % normalise moments for scale invariance
    cm = cm/(cm_00^(1+(p+q)/2));
    
end

% calculate center of mass
function [x_bar, y_bar] = centerOfMass(image,xgrid,ygrid)

    eps = 10^(-6); % very small constant 
    
    x_bar = sum(sum((xgrid.*image)))/(sum(image(:))+eps);
    y_bar = sum(sum((ygrid.*image)))/(sum(image(:))+eps);

end

