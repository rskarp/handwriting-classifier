% Convert numeric feature data to matrix & get class labels
% INPUTS:
%   - features: 1xN cell array of structs. Each element contains extracted
% features for one letter.
%   - chars: string of unique characters present in features. e.g. 'ABCDEFG'
%   - normalize: boolean flag indicating whether or not to normalize the
% columns of the output feature data matrix
%   - equalClasses: boolean flag indicating whether or not to limit the
%   number of letters per class so the data contains an equal number of
%   letters per class
% OUTPUTS:
%   - data: NxF matrix of feature data. N = # observations, F = # features
% (hard-coded as 19 for now)
%   - labels: 1xN array of integers indicating the class label for each
% observation. The integers correspond to the index of the character in
% chars. e.g. Given label = 2 and chars = 'ABC', the class would be 'B'
%   - letters: 1xN string of characters indicating the class for each
% observation. Contains the letters corresponding to the integers in the
% labels array
%   - labelMatrix: NxC matrix, where C is the number of classes (length of
% chars input). Each row contains a 1 at the index of the corresponding
% class, and the rest of the row is 0's. MATLAB's patternnet requires
% labels in this format.
function [data, labels, letters, labelMatrix] = convertFeaturesToMatrix(features, chars, normalize, equalClasses)
    NUM_FEATURES = 21; % Update this if our extracted features change
    NUM_POINTS = length(features);

    % Determine how many letters of each class should be returned
    if equalClasses == 1
        trainLetters = cellfun(@(x) x.Letter,features);
        minCount = intmax;
        for i = 1:length(chars)
            c = chars(i);
            s = sum(cellfun(@(x) x == c,num2cell(trainLetters)));
            minCount = min([minCount,s]);
        end
        letterCounts = zeros(size(chars));
        NUM_POINTS = minCount * length(chars);
    end

    data = zeros([NUM_POINTS,NUM_FEATURES]);
    labels = zeros([1,NUM_POINTS]);
    letters = strings([1,NUM_POINTS]);
    labelMatrix = zeros([NUM_POINTS,length(chars)]);

    idx = 1;
    for i = 1:length(features)
        obj = features{i};
        class = strfind(chars,obj.Letter);

        % Increment how many of this letter we've stored
        if equalClasses == 1 && ~isempty(class)
            letterCounts(class) = letterCounts(class) + 1;
        end

        % Store the letter information if we need more of this letter
        if equalClasses ~= 1 || letterCounts(class) <= minCount
            row = [obj.Area, obj.Centroid(1), obj.Centroid(2), obj.MajorAxisLength,...
                obj.MinorAxisLength, obj.Eccentricity, obj.Orientation, obj.ConvexArea,...
                obj.Area/obj.ConvexArea,obj.MajorAxisLength/obj.MinorAxisLength,...
                obj.Circularity, obj.Solidity, obj.Perimeter, obj.HuMoments];
            letters(idx) = obj.Letter;
            data(idx,:) = row;
            if ~isempty(class)
                labels(idx) = class;
                labelMatrix(idx,class) = 1;
            end
            idx = idx+1;
        end
    end
    
    % Normalize feature columns
    if normalize == 1
        data = normc(data);
    end
end