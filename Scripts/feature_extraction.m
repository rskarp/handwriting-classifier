% Data Conditioning & Feature extraction code
close all;
clear;


% SCRIPT SETTINGS
% File containing metadata (image file name, label)
MetadataFilePath = 'Dataset/written_name_train_v2.csv';
% Path to folder containing image data
ImageDataPath = 'Dataset/train_v2/train/';
% .mat file to save feature data to
OutputFeaturesFilePath = 'Features/train_features.mat';
% Boolean: Set to 1 to display plots for every image, else 0
showPlots = 1;
% Boolean: Set to 1 to save features to .mat file, else 0
saveFeatures = 1;
% Boolean: Set to 1 if training data (labels are known), else set to 0
train = 1;


% Read in Training Metadata (filename, trueName)
trainMeta = readmatrix(MetadataFilePath,'OutputType','string');
% Remove column headers
trainMeta = trainMeta(2:length(trainMeta),:);

% Only use a few image files for now
data = trainMeta(1:5,:);

% Initialize variables
featuresData = {};
numFeatureObjs = 0;

% Iterate through training images
for i = 1:length(data)
    filename = data(i,1);
    % Read original image
    I = imread(ImageDataPath+filename);
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
    Ithresh = Ierode < 135;

    % Plot Images
    if showPlots
        figure; 
        tiledlayout(4,2,"TileSpacing","compact");
        nexttile, imshow(I), title('Original');
        nexttile, imshow(Igray), title('RGB converted to gray');
        nexttile, imshow(Ienhanced), title('Contrast Enhanced');
        nexttile, imshow(Iopen), title('Morphologically Opened');
        nexttile, imshow(Ierode), title('Morphologically Eroded');
        nexttile, imshow(Ithresh), title('Binary Thresholded');
    end

    % Extract objects and properties with regionprops
    cc = bwconncomp(Ithresh*255,4);
    allObjects = regionprops(cc,'Area','Centroid','Circularity', ...
        'ConvexArea','Eccentricity','Image','MajorAxisLength', ...
        'MinorAxisLength','Orientation','Perimeter', 'PixelIdxList', ...
        'Solidity'); 
    
    % Filter out small & horizontal(merged letters) objects
    objects = allObjects(arrayfun(@(x) x.ConvexArea > 20 && x.ConvexArea < 300 &&...
        (abs(x.Orientation) > 10 || x.MajorAxisLength/x.MinorAxisLength < 2),allObjects));

    % Cluster objects into groups in the same line using basic sequential
    % clustering.
    % Note: this clustering step is specific to our dataset so we can
    % identify the handwritten name in the image. This likely prevents
    % generalizability to other images with multiple lines of handwritten
    % text.
    numGroups = 1;
    groupCenters = zeros([length(objects) 2]);
    groupCenters(1,:) = objects(1).Centroid;
    groupCount = zeros([1 length(objects)]);
    groupArea = zeros([1 length(objects)]);
    for ii=1:length(objects)
        isGroupMember = 0;
        objLoc = objects(ii).Centroid;
        for k=1:numGroups
            c = groupCenters(k,:);
            isSimilarCol = abs(objLoc(1)-c(1)) < groupCount(k)*20; % x
            isSimilarRow = abs(objLoc(2)-c(2)) < 10; % y
            if (isSimilarRow && isSimilarCol)
                % Group exists, increment group count, adjust group centroid
                % using weighted average
                weights = [groupCount(k); 1];
                groupCenters(k,:) = sum([c;objLoc].*weights)/sum(weights);
                groupCount(k) = groupCount(k)+1;
                groupArea(k) = groupArea(k)+objects(ii).ConvexArea;
                isGroupMember = 1;
                objects(ii).Group = k;
                break;
            end
        end
        if (isGroupMember==0)
            % New group found
            numGroups = numGroups + 1;
            groupCenters(numGroups,:) = objLoc;
            groupCount(numGroups) = 1;
            objects(ii).Group = numGroups;
            groupArea(numGroups) = objects(ii).ConvexArea;
        end
    end
    
    % Find cluster with largest pixel area
    [numObj,k] = max(groupArea);
    % Get all objects in the largest cluster
    finalObjects = objects(arrayfun(@(x) x.Group==k ,objects));

    % Display final objects in one image
    objectsImage = false(size(Ithresh));
    for j=1:length(finalObjects)
        objectsImage(finalObjects(j).PixelIdxList) = true;
    end
    if showPlots
        nexttile, imshow(objectsImage), title('Objects');
    end

    % If training (label is known), get label
    if train
        identity = data(i,2);
        identity = replace(erasePunctuation(identity)," ","");
    else
        identity = "";
    end

    % If label is known (e.g. training) ensure  # objects = # characters
    if ~isempty(identity)
        % If # objects doesn't match # characters in label, don't use this
        % bad data for training
        if length(finalObjects) ~= strlength(identity)
            finalObjects = [];
        end
        % Plot letters
        if showPlots && ~isempty(finalObjects)
            figure;
            tiledlayout(1,length(finalObjects),"TileSpacing","compact");
            for j=1:length(finalObjects)
                objectsImage = padarray(finalObjects(j).Image,[1 1],0,'both');
                nexttile, imshow(objectsImage), title(identity{1}(j));
            end
        end
    end

    % Store feature data
    for j=1:length(finalObjects)
        numFeatureObjs = numFeatureObjs + 1;
        obj = finalObjects(j);
        objectsImage = false(size(Ithresh));
        objectsImage(obj.PixelIdxList) = true;
        % Add metadata
        obj.FullImage = objectsImage;
        obj.LetterImage = padarray(obj.Image,[1 1],0,'both');
        obj.Filename = filename;
        obj.Name = data(i,2);
        if ~isempty(identity) && length(finalObjects) == strlength(identity)
            obj.Letter = identity{1}(j);
        else
            obj.Letter = '';
        end
        % Remove unneeded attributes
        obj = rmfield(obj,{'PixelIdxList','Group','Image'});
        featuresData{numFeatureObjs} = obj;
    end
end

% Save features to file
if saveFeatures
    save(OutputFeaturesFilePath,'featuresData');
end
