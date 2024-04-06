
% Function to convert grayscale image to a bianary image

function B = im2bin(A)

    B = imbinarize(A,'adaptive','ForegroundPolarity','dark','Sensitivity',0.45);
    B = not(B);

end
