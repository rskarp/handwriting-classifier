% Load all features from all files in the folder for a given dataset.
% Input: dataset - can be "train", "validation", or "test"
% Output: data - 1xn cell array of all feature objects
function data = load_features(dataset)
    if dataset ~= "train" && dataset ~= "validation" && dataset ~= "test"
        fprintf('dataset must have value "train", "validation", or "test"\n');
        return;
    end

    featureDir = dir(strcat('Features/', dataset, '/*.mat'));

    % Initialize data array. Assume 100,000 data points per feature file (since
    % that's how we saved them in feature_extraction.m)
    data = cell([1 100000*length(featureDir)]);
    
    % Load all data from all feature files in the folder
    curIdx = 1;
    for i=1:length(featureDir)
        newData = load(strcat(featureDir(i).folder,'/',featureDir(i).name)).data;
        
        data(curIdx:curIdx+length(newData)-1) = newData;
        curIdx = curIdx + length(newData);
    end
    
    % We assumed all files have 100,000 data points. Remove any empty cells at
    % the end since the last file may not have 100,000 points.
    data = data(~cellfun(@isempty,data));
end
