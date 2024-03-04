% Read the image
img_dir = dir("Dataset/train_v2/train/*.jpg");

for im=10 %length(img(dir))
    image = imread(strcat(img_dir(im).folder,'/',img_dir(im).name));

    figure(im); % For testing, generate a new figure for each image
    [~,stats] = precondition(image,1); % Precondition image with grayscale & thresholding


    % Debug section - code to see bounding boxes
    % % imshow(image);
    % % hold on;
    % % 
    % % % Loop through each region and draw bounding boxes around handwritten data
    % % for i = 1:length(stats)
    % %     boundingBox = S_sorted(i).BoundingBox;
    % %     rectangle('Position', [boundingBox(1), boundingBox(2), boundingBox(3), boundingBox(4)],...
    % %               'EdgeColor', 'r', 'LineWidth', 2);
    % % end
    % % 
    % % hold off;

    %% crop to each bounding box to check for words with length = spreadsheet
    
    % Import CSV data
    trainMeta = readmatrix('Dataset/written_name_train_v2.csv','OutputType','string');
    
    % Remove column headers
    trainMeta = trainMeta(2:length(trainMeta),:);
    
    % Calculate length of name
    name_length = length(trainMeta{im,2});


for i = 1:length(stats)

    % Crop the region from the original image
    cropped_image = imcrop(image, stats(i).BoundingBox);
    
    %figure(); imshow(cropped_image)

    [~,crop_stats] = precondition(cropped_image,0);

    % More debug code
    % % hold on;
    % % 
    % % % Loop through each region and draw bounding boxes around handwritten data
    % % for m = 1:length(crop_stats)
    % %     boundingBox = crop_stats(m).BoundingBox;
    % %     rectangle('Position', [boundingBox(1), boundingBox(2), boundingBox(3), boundingBox(4)],...
    % %               'EdgeColor', 'r', 'LineWidth', 2);
    % % end
    % % 
    % % hold off;

    % Check for character count in BB to match name length
    if length(crop_stats)==name_length 
        for j = 1:length(crop_stats)

            % Increase bounding box by 10% 
            old_dims = crop_stats(j).BoundingBox(3:4);
            new_dims = round(crop_stats(j).BoundingBox(3:4) * 1.2);

            xy_shift = round((new_dims - old_dims)/2);

            crop_stats(j).BoundingBox(1:2) = crop_stats(j).BoundingBox(1:2) - xy_shift;
            crop_stats(j).BoundingBox(3:4) = new_dims;

            % Crop to characters
            char_crop = imcrop(cropped_image, crop_stats(j).BoundingBox);
        
            % Resize the cropped image to a standard size (e.g., 28x28)
            resized_image = imresize(char_crop, [28, 28]);
        
            % Store the resized image
            resized_images{i} = resized_image;
    
            % Display the resized image
            subplot(1, name_length, j);
            imshow(resized_image);
            title([trainMeta{im,2}(j)]);
        end

    end
end

% Save or do further processing with the resized images as needed

end


function [binary_image, stats] = precondition(image, iter)
% binary_image - returns the binary image in its final form
% stats - returns the regionprops data for the image
% image - the image that is being sent through preconditioning
% iter - iteration to determine if we need to dilate or not, value should
%           be 0 or 1
   
    % Convert the image to grayscale if it's in RGB
    if size(image, 3) == 3
        gray_image = rgb2gray(image);
    else
        gray_image = image;
    end
    
    % Thresholding to binarize the image
    threshold = graythresh(gray_image);
    binary_image = imbinarize(gray_image, threshold);
    
    % Fill small gaps in the binary image, leaving if necessary later
    %binary_image = imfill(binary_image, 'holes');
    
    % Remove small objects (noise)
    binary_image = bwareaopen(binary_image, 20); % Adjust the parameter according to your image
    
    % Invert the binary image so that handwritten regions are white
    binary_image = ~binary_image;
    
    if iter
        se = strel('line', 15, 0); % Adjust the size of the structuring element as needed
        binary_image_cleaned = imdilate(binary_image, se);
        
        %imshow(binary_image_cleaned) % Debug line
        
        CC = bwconncomp(binary_image_cleaned);
        
        % Find bounding box of handwritten regions
        stats = regionprops(CC);
    else
        stats = regionprops(binary_image);
    end
    


end
