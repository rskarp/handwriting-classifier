% Feature extraction code
close all;
clear;

% Read in Training Metadata (filename, trueName)
trainMeta = readmatrix('Dataset/written_name_train_v2.csv','OutputType','string');
% Remove column headers
trainMeta = trainMeta(2:length(trainMeta),:);

% Only use first 3 files for now
first3 = trainMeta(1:3,:);

% Iterate through training images
trainDataPath = 'Dataset/train_v2/train/';
for i = 1:length(first3)
    filename = first3(i,1);
    I = imread(trainDataPath+filename);
    figure; imshow(I);
end
