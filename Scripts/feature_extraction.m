% Data Conditioning & Feature extraction code
close all;
clear;

% Read in Training Metadata (filename, trueName)
trainMeta = readmatrix('Dataset/written_name_train_v2.csv','OutputType','string');
% Remove column headers
trainMeta = trainMeta(2:length(trainMeta),:);

% Only use first 3 files for now
first4 = trainMeta(1:4,:);

% Iterate through training images
showPlots=1;
trainDataPath = 'Dataset/train_v2/train/';
for i = 1:length(first4)
    filename = first4(i,1);
    % Read original image
    I = imread(trainDataPath+filename);
    % Convert to bw so it can be passed to imadjust
    Igray = rgb2gray(I);
    % Enhance contrast using imadjust
    Ienhanced = imadjust(Igray);
    % Morphological opening to remove small background noises
    se = strel('square',3);
    Iopen = imopen(Ienhanced,se);
    % Morphological erosion to sharpen edges
    se = strel('square',2);
    Ierode = imerode(Iopen,se);
    % Binarize image so we can use regionprops
    Ithresh = Ierode > 128;

    % Plot Images
    if showPlots
        figure; 
        tiledlayout(3,2,"TileSpacing","compact");
        nexttile, imshow(I), title('Original');
        nexttile, imshow(Igray), title('RGB converted to gray');
        nexttile, imshow(Ienhanced), title('Contrast Enhanced');
        nexttile, imshow(Iopen), title('Morphologically Opened');
        nexttile, imshow(Ierode), title('Morphologically Eroded');
        nexttile, imshow(Ithresh), title('Binary Thresholded');
    end
    
    % TO DO:
    % Constant image height
    % Extract objects/spaces - remove hyphens, typed letters, small objects
    % Extract letter features: regionpros, Hu moments
end
