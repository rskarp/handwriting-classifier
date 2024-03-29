% Convert numeric feature data to matrix & get class labels
% INPUTS:
%   - features: 1xN cell array of structs. Each element contains extracted
% features for one letter.
%   - chars: string of unique characters present in features. e.g. 'ABCDEFG'
%   - normalize: boolean flag indicating whether or not to normalize the
% columns of the output feature data matrix
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
function [data, labels, letters, labelMatrix] = convertFeaturesToMatrix(features, chars, normalize)
    NUM_FEATURES = 19; % Update this if our extracted features change
    data = zeros([length(features),NUM_FEATURES]);
    labels = zeros([1,length(features)]);
    letters = strings([1,length(features)]);
    labelMatrix = zeros([length(features),length(chars)]);
    for i = 1:length(features)
        obj = features{i};
        row = [obj.Area, obj.Centroid(1), obj.Centroid(2), obj.MajorAxisLength,...
            obj.MinorAxisLength, obj.Eccentricity, obj.Orientation, obj.ConvexArea,...
            obj.Circularity, obj.Solidity, obj.Perimeter, obj.HuMoments];
        class = strfind(chars,obj.Letter);
        letters(i) = obj.Letter;
        data(i,:) = row;
        if ~isempty(class)
            labels(i) = class;
            labelMatrix(i,class) = 1;
        end
    end
    % Normalize feature columns
    if normalize == 1
        data = normc(data);
    end
end